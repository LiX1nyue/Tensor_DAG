#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace {

struct Timing {
    std::string duration;
    std::string self_duration;
};

using TimingTable = std::unordered_map<std::string, std::vector<Timing>>;
struct TraceStart {
    int64_t sequence;
    std::string timestamp;
};
struct TraceEntry {
    std::string name;
    int64_t sequence;
    std::string timestamp;
    std::size_t csv_index = 0;
    bool used = false;
};
using StartTable = std::vector<TraceEntry>;
using NodeKey = std::pair<std::string, int64_t>;

std::string trim(const std::string& value) {
    std::size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) ++first;
    std::size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) --last;
    return value.substr(first, last - first);
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (c == ',' && !quoted) {
            fields.push_back(field);
            field.clear();
        } else {
            field += c;
        }
    }
    fields.push_back(field);
    return fields;
}

TimingTable load_timings(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open operator details: " + path);

    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty operator details: " + path);
    const auto header = parse_csv_line(line);
    std::unordered_map<std::string, std::size_t> column;
    for (std::size_t i = 0; i < header.size(); ++i) column[trim(header[i])] = i;

    if (!column.count("Name") || !column.count("Host Self Duration(us)") ||
        !column.count("Host Total Duration(us)")) {
        throw std::runtime_error("operator details is missing required columns");
    }

    TimingTable timings;
    const auto name_col = column["Name"];
    const auto self_col = column["Host Self Duration(us)"];
    const auto total_col = column["Host Total Duration(us)"];
    while (std::getline(input, line)) {
        const auto fields = parse_csv_line(line);
        if (fields.size() <= std::max({name_col, self_col, total_col})) continue;
        timings[fields[name_col]].push_back({fields[total_col], fields[self_col]});
    }
    return timings;
}

class JsonStream {
public:
    explicit JsonStream(std::ifstream& input) : input_(input) {}

    char peek() {
        fill();
        return position_ == buffer_.size() ? '\0' : buffer_[position_];
    }

    char get() {
        fill();
        if (position_ == buffer_.size()) return '\0';
        ++absolute_position_;
        return buffer_[position_++];
    }

    std::size_t position() const { return absolute_position_; }

    void copy_to(std::ostream& output, std::size_t end) {
        while (absolute_position_ < end) {
            fill();
            const auto count = std::min(end - absolute_position_, buffer_.size() - position_);
            output.write(buffer_.data() + position_, static_cast<std::streamsize>(count));
            position_ += count;
            absolute_position_ += count;
        }
    }

private:
    void fill() {
        if (position_ < buffer_.size() || input_.eof()) return;
        buffer_.resize(1 << 16);
        input_.read(buffer_.data(), static_cast<std::streamsize>(buffer_.size()));
        buffer_.resize(static_cast<std::size_t>(input_.gcount()));
        position_ = 0;
    }

    std::ifstream& input_;
    std::string buffer_;
    std::size_t position_ = 0;
    std::size_t absolute_position_ = 0;
};

void copy_string(JsonStream& json, std::ostream& output, std::string* decoded = nullptr) {
    if (json.get() != '"') throw std::runtime_error("expected JSON string");
    output.put('"');
    bool escaped = false;
    while (true) {
        const char c = json.get();
        if (c == '\0') throw std::runtime_error("unterminated JSON string");
        output.put(c);
        if (escaped) {
            if (decoded) {
                if (c == 'n') decoded->push_back('\n');
                else if (c == 'r') decoded->push_back('\r');
                else if (c == 't') decoded->push_back('\t');
                else decoded->push_back(c);
            }
            escaped = false;
        } else if (c == '\\') {
            escaped = true;
        } else if (c == '"') {
            return;
        } else if (decoded) {
            decoded->push_back(c);
        }
    }
}

std::string json_number(const std::string& value) {
    try {
        std::size_t consumed = 0;
        const auto number = std::stod(value, &consumed);
        if (consumed == value.size()) {
            std::ostringstream output;
            output.precision(17);
            output << number;
            return output.str();
        }
    } catch (const std::exception&) {
    }
    return value;
}

void copy_value(JsonStream& json, std::ostream& output) {
    char c = json.peek();
    if (c == '"') {
        copy_string(json, output);
    } else if (c == '{' || c == '[') {
        const char closing = c == '{' ? '}' : ']';
        output.put(json.get());
        while (true) {
            c = json.peek();
            if (c == '\0') throw std::runtime_error("unterminated JSON value");
            if (std::isspace(static_cast<unsigned char>(c))) {
                output.put(json.get());
            } else if (c == '"') {
                copy_string(json, output);
            } else if (c == '{' || c == '[') {
                copy_value(json, output);
            } else if (c == closing) {
                output.put(json.get());
                return;
            } else {
                output.put(json.get());
            }
        }
    } else {
        while ((c = json.peek()) != '\0' && c != ',' && c != ']' && c != '}' &&
               !std::isspace(static_cast<unsigned char>(c))) {
            output.put(json.get());
        }
    }
}

// Discard unrelated trace fields without retaining a full trace_view in memory.
class NullBuffer : public std::streambuf {
    int_type overflow(int_type c) override { return traits_type::not_eof(c); }
};

void skip_space(JsonStream& json) {
    while (std::isspace(static_cast<unsigned char>(json.peek()))) json.get();
}

StartTable load_starts(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open trace view: " + path);
    JsonStream json(input);
    NullBuffer buffer;
    std::ostream discard(&buffer);
    StartTable starts;
    skip_space(json);
    if (json.get() != '[') throw std::runtime_error("expected trace_view event array");
    skip_space(json);
    while (json.peek() != ']') {
        if (json.get() != '{') throw std::runtime_error("expected trace event object");
        std::string name, category, phase, timestamp;
        int64_t sequence = -1;
        skip_space(json);
        while (json.peek() != '}') {
            std::string key;
            copy_string(json, discard, &key);
            skip_space(json);
            if (json.get() != ':') throw std::runtime_error("expected trace event colon");
            skip_space(json);
            std::string* value = key == "name" ? &name : key == "cat" ? &category :
                                 key == "ph" ? &phase : key == "ts" ? &timestamp : nullptr;
            if (value && json.peek() == '"') {
                copy_string(json, discard, value);
            } else if (key == "ts") {
                std::ostringstream text;
                copy_value(json, text);
                timestamp = text.str();
            } else if (key == "args") {
                if (json.get() != '{') throw std::runtime_error("expected trace event args object");
                skip_space(json);
                while (json.peek() != '}') {
                    std::string arg_name;
                    copy_string(json, discard, &arg_name);
                    skip_space(json);
                    if (json.get() != ':') throw std::runtime_error("expected trace args colon");
                    skip_space(json);
                    if (arg_name == "Sequence number") {
                        std::ostringstream number;
                        copy_value(json, number);
                        try { sequence = std::stoll(number.str()); } catch (...) { sequence = -1; }
                    } else {
                        copy_value(json, discard);
                    }
                    skip_space(json);
                    if (json.peek() == '}') break;
                    if (json.get() != ',') throw std::runtime_error("expected trace args separator");
                    skip_space(json);
                }
                json.get();
            } else {
                copy_value(json, discard);
            }
            skip_space(json);
            if (json.peek() == '}') break;
            if (json.get() != ',') throw std::runtime_error("expected trace field separator");
            skip_space(json);
        }
        json.get();
        if (category == "cpu_op" && phase == "X" && !name.empty() &&
            name.rfind("ProfilerStep#", 0) != 0 &&
            !timestamp.empty() && timestamp != "null") {
            // Keep the original decimal digits: epoch microseconds lose
            // fractional precision if converted to double.
            starts.push_back({name, sequence, timestamp});
        }
        skip_space(json);
        if (json.peek() == ']') break;
        if (json.get() != ',') throw std::runtime_error("expected trace event separator");
        skip_space(json);
    }
    return starts;
}

std::vector<NodeKey> load_host_keys(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open host trace: " + path);
    JsonStream json(input);
    NullBuffer buffer;
    std::ostream discard(&buffer);
    std::vector<NodeKey> keys;
    bool in_string = false;
    bool escaped = false;
    while (json.peek() != '\0') {
        if (!in_string && json.peek() == '"') {
            std::string key;
            copy_string(json, discard, &key);
            if (key == "nodes") {
                skip_space(json);
                if (json.get() != ':') throw std::runtime_error("expected nodes property value");
                skip_space(json);
                if (json.get() != '[') throw std::runtime_error("nodes property is not an array");
                skip_space(json);
                while (json.peek() != ']') {
                    if (json.get() != '{') throw std::runtime_error("invalid item in nodes array");
                    std::string name;
                    int64_t sequence = -1;
                    skip_space(json);
                    while (json.peek() != '}') {
                        std::string field;
                        copy_string(json, discard, &field);
                        skip_space(json);
                        if (json.get() != ':') throw std::runtime_error("expected node field colon");
                        skip_space(json);
                        if (field == "name" && json.peek() == '"') {
                            copy_string(json, discard, &name);
                        } else if (field == "attrs") {
                            if (json.get() != '[') throw std::runtime_error("expected node attrs array");
                            skip_space(json);
                            while (json.peek() != ']') {
                                if (json.get() != '{') throw std::runtime_error("invalid node attribute");
                                std::string attr_name;
                                skip_space(json);
                                while (json.peek() != '}') {
                                    std::string attr_field;
                                    copy_string(json, discard, &attr_field);
                                    skip_space(json);
                                    if (json.get() != ':') throw std::runtime_error("expected attribute colon");
                                    skip_space(json);
                                    if (attr_field == "name") {
                                        copy_string(json, discard, &attr_name);
                                    } else if (attr_field == "value" && attr_name == "seq_id") {
                                        std::ostringstream number;
                                        copy_value(json, number);
                                        try { sequence = std::stoll(number.str()); } catch (...) { sequence = -1; }
                                    } else {
                                        copy_value(json, discard);
                                    }
                                    skip_space(json);
                                    if (json.peek() == '}') break;
                                    if (json.get() != ',') throw std::runtime_error("expected attribute separator");
                                    skip_space(json);
                                }
                                json.get();
                                skip_space(json);
                                if (json.peek() == ']') break;
                                if (json.get() != ',') throw std::runtime_error("expected attrs separator");
                                skip_space(json);
                            }
                            json.get();
                        } else {
                            copy_value(json, discard);
                        }
                        skip_space(json);
                        if (json.peek() == '}') break;
                        if (json.get() != ',') throw std::runtime_error("expected node field separator");
                        skip_space(json);
                    }
                    json.get();
                    if (!name.empty() && sequence >= 0) keys.emplace_back(name, sequence);
                    skip_space(json);
                    if (json.peek() == ']') break;
                    if (json.get() != ',') throw std::runtime_error("expected node array separator");
                    skip_space(json);
                }
                return keys;
            }
        } else {
            const char c = json.get();
            if (in_string) {
                if (escaped) escaped = false;
                else if (c == '\\') escaped = true;
                else if (c == '"') in_string = false;
            } else if (c == '"') {
                in_string = true;
            }
        }
    }
    throw std::runtime_error("host trace has no nodes array");
}

int64_t infer_sequence_offset(const std::vector<NodeKey>& host_keys,
                              const StartTable& starts) {
    std::unordered_map<std::string, std::vector<int64_t>> trace_sequences;
    for (const auto& entry : starts) {
        if (entry.sequence >= 0) trace_sequences[entry.name].push_back(entry.sequence);
    }
    std::unordered_map<std::string, std::size_t> trace_index;
    std::map<int64_t, std::size_t> offsets;
    for (const auto& [name, host_sequence] : host_keys) {
        const auto found = trace_sequences.find(name);
        if (found == trace_sequences.end()) continue;
        auto& ti = trace_index[name];
        if (ti < found->second.size()) {
            ++offsets[host_sequence - found->second[ti]];
            ++ti;
        }
    }
    if (offsets.empty()) throw std::runtime_error("cannot infer sequence-number offset");
    const auto best = std::max_element(offsets.begin(), offsets.end(),
        [](const auto& left, const auto& right) { return left.second < right.second; });
    if (best->second < 2) throw std::runtime_error("too few events to infer sequence-number offset");
    return best->first;
}

using TraceLookup = std::unordered_map<std::string,
    std::unordered_map<int64_t, std::vector<std::size_t>>>;

TraceLookup index_starts(const StartTable& starts, int64_t offset) {
    TraceLookup lookup;
    for (std::size_t i = 0; i < starts.size(); ++i) {
        if (starts[i].sequence >= 0)
            lookup[starts[i].name][starts[i].sequence + offset].push_back(i);
    }
    return lookup;
}

void map_csv_rows(StartTable& starts, const TimingTable& timings) {
    std::unordered_map<std::string, std::size_t> indexes;
    for (auto& entry : starts) {
        const auto found = timings.find(entry.name);
        const auto index = indexes[entry.name]++;
        if (found == timings.end() || index >= found->second.size())
            throw std::runtime_error("trace/CSV row count mismatch for " + entry.name);
        entry.csv_index = index;
    }
}

void copy_attrs(JsonStream& json, std::ostream& output, int64_t& sequence) {
    if (json.get() != '[') throw std::runtime_error("expected node attrs array");
    output.put('[');
    std::string whitespace;
    skip_space(json);
    while (json.peek() != ']') {
        if (json.get() != '{') throw std::runtime_error("expected node attribute object");
        output.put('{');
        std::string attr_name;
        std::string pending;
        skip_space(json);
        while (json.peek() != '}') {
            const char c = json.peek();
            if (std::isspace(static_cast<unsigned char>(c))) {
                const auto ws = json.get();
                output.put(ws);
                pending.push_back(ws);
            } else if (c == '"') {
                std::string key;
                copy_string(json, output, &key);
                while (std::isspace(static_cast<unsigned char>(json.peek()))) output.put(json.get());
                if (json.get() != ':') throw std::runtime_error("expected attribute field colon");
                output.put(':');
                while (std::isspace(static_cast<unsigned char>(json.peek()))) output.put(json.get());
                if (key == "name") {
                    copy_string(json, output, &attr_name);
                } else if (key == "value" && attr_name == "seq_id") {
                    std::ostringstream number;
                    copy_value(json, number);
                    try { sequence = std::stoll(number.str()); } catch (...) { sequence = -1; }
                    output << number.str();
                } else {
                    copy_value(json, output);
                }
                pending.clear();
            } else if (c == ',') {
                output << pending;
                pending.clear();
                output.put(json.get());
            } else {
                throw std::runtime_error("invalid node attribute object");
            }
        }
        json.get();
        output << pending << '}';
        skip_space(json);
        if (json.peek() == ']') break;
        if (json.get() != ',') throw std::runtime_error("expected attrs separator");
        output.put(',');
        skip_space(json);
    }
    json.get();
    output.put(']');
}

bool copy_node(JsonStream& json, std::ostream& output, const TimingTable& timings,
               StartTable& starts, const TraceLookup& trace_lookup,
               std::unordered_map<std::string, std::size_t>& fallback_indexes,
               const std::string& newline, int& matched, int& start_matched) {
    if (json.get() != '{') throw std::runtime_error("expected node object");
    output.put('{');
    std::string name;
    std::string pending_whitespace;
    int64_t sequence = -1;
    bool has_name = false;
    bool has_field = false;

    while (true) {
        char c = json.peek();
        if (c == '\0') throw std::runtime_error("unterminated node object");
        if (std::isspace(static_cast<unsigned char>(c))) {
            pending_whitespace.push_back(json.get());
        } else if (c == '}') {
            json.get();
            const auto found = timings.find(name);
            std::size_t trace_index = starts.size();
            if (has_name && sequence >= 0) {
                const auto by_name = trace_lookup.find(name);
                if (by_name != trace_lookup.end()) {
                    const auto by_sequence = by_name->second.find(sequence);
                    if (by_sequence != by_name->second.end()) {
                        for (const auto candidate : by_sequence->second) {
                            if (!starts[candidate].used) {
                                trace_index = candidate;
                                break;
                            }
                        }
                    }
                }
            }
            if (trace_index == starts.size() && has_name && sequence < 0) {
                for (std::size_t i = fallback_indexes[name]; i < starts.size(); ++i) {
                    if (!starts[i].used && starts[i].sequence < 0 && starts[i].name == name) {
                        trace_index = i;
                        fallback_indexes[name] = i + 1;
                        break;
                    }
                }
            }
            const bool has_trace = trace_index < starts.size();
            if (has_trace) starts[trace_index].used = true;
            const bool has_timing = has_trace && found != timings.end() &&
                                    starts[trace_index].csv_index < found->second.size();
            std::string fields;
            if (has_trace) {
                fields = "\"start_ts\": " + json_number(starts[trace_index].timestamp);
                ++start_matched;
            }
            if (has_timing) {
                const auto& timing = found->second[starts[trace_index].csv_index];
                if (!fields.empty()) fields += ", ";
                fields += "\"duration\": " + timing.duration +
                          ", \"self_duration\": " + timing.self_duration;
                ++matched;
            }
            if (!fields.empty()) {
                if (pending_whitespace.find('\n') == std::string::npos) {
                    output << pending_whitespace << ", " << fields;
                } else {
                    const auto indent_start = pending_whitespace.rfind('\n');
                    const auto close_indent = pending_whitespace.substr(indent_start + 1);
                    if (has_field) output.put(',');
                    output << newline << close_indent << "  " << fields
                           << pending_whitespace;
                }
            } else {
                output << pending_whitespace;
            }
            output.put('}');
            return has_trace;
        } else if (c == ',') {
            output << pending_whitespace;
            pending_whitespace.clear();
            output.put(json.get());
            has_field = true;
        } else if (c == '"') {
            output << pending_whitespace;
            pending_whitespace.clear();
            std::string key;
            copy_string(json, output, &key);
            while (std::isspace(static_cast<unsigned char>(json.peek()))) output.put(json.get());
            const char colon = json.get();
            if (colon != ':') throw std::runtime_error("expected colon after object key");
            output.put(colon);
            while (std::isspace(static_cast<unsigned char>(json.peek()))) output.put(json.get());
            const bool is_name = key == "name";
            std::string value;
            if (is_name && json.peek() == '"') {
                copy_string(json, output, &value);
                name = value;
                has_name = true;
            } else if (key == "attrs") {
                copy_attrs(json, output, sequence);
            } else {
                copy_value(json, output);
            }
            has_field = true;
        } else {
            throw std::runtime_error("invalid node object");
        }
    }
}

void bind_stream(const std::string& host_path, const std::string& output_path,
                 const TimingTable& timings, StartTable& starts,
                 const TraceLookup& trace_lookup,
                 int& total, int& matched, int& start_matched) {
    std::ifstream input(host_path, std::ios::binary);
    std::ofstream output(output_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open host trace: " + host_path);
    if (!output) throw std::runtime_error("cannot open output: " + output_path);
    JsonStream json(input);
    std::unordered_map<std::string, std::size_t> indexes;
    std::string recent;
    std::string newline = "\n";
    bool in_string = false;
    bool escaped = false;
    bool found_nodes = false;

    while (json.peek() != '\0') {
        const char c = json.peek();
        if (!in_string && c == '"') {
            std::string key;
            copy_string(json, output, &key);
            if (key == "nodes") {
                found_nodes = true;
                while (std::isspace(static_cast<unsigned char>(json.peek()))) output.put(json.get());
                if (json.get() != ':') throw std::runtime_error("expected nodes property value");
                output.put(':');
                while (std::isspace(static_cast<unsigned char>(json.peek()))) output.put(json.get());
                if (json.get() != '[') throw std::runtime_error("nodes property is not an array");
                output.put('[');
                while (true) {
                    while (std::isspace(static_cast<unsigned char>(json.peek()))) output.put(json.get());
                    if (json.peek() == ']') {
                        output.put(json.get());
                        break;
                    }
                    if (json.peek() == ',') {
                        output.put(json.get());
                        continue;
                    }
                    if (json.peek() != '{') throw std::runtime_error("invalid item in nodes array");
                    ++total;
                    copy_node(json, output, timings, starts, trace_lookup, indexes,
                              newline, matched, start_matched);
                }
                continue;
            }
            in_string = false;
        } else {
            const char copied = json.get();
            output.put(copied);
            if (in_string) {
                if (escaped) escaped = false;
                else if (copied == '\\') escaped = true;
                else if (copied == '"') in_string = false;
            } else if (copied == '"') {
                in_string = true;
            }
            recent.push_back(copied);
            if (recent.size() > 2) recent.erase(0, recent.size() - 2);
            if (recent == "\r\n") newline = "\r\n";
        }
    }
    if (!found_nodes) throw std::runtime_error("host trace has no nodes array");
    if (!output) throw std::runtime_error("failed while writing output");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string host, details, output, trace_view;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + option);
            if (option == "--host") host = argv[++i];
            else if (option == "--operator-details") details = argv[++i];
            else if (option == "--trace-view") trace_view = argv[++i];
            else if (option == "--output") output = argv[++i];
            else throw std::runtime_error("unknown option: " + option);
        }
        if (host.empty() || details.empty() || output.empty())
            throw std::runtime_error("usage: bind_operator_details --host HOST --operator-details CSV [--trace-view TRACE] --output OUTPUT");

        const auto timings = load_timings(details);
        auto starts = trace_view.empty() ? StartTable{} : load_starts(trace_view);
        const auto host_keys = trace_view.empty() ? std::vector<NodeKey>{} : load_host_keys(host);
        const int64_t offset = trace_view.empty() ? 0 : infer_sequence_offset(host_keys, starts);
        map_csv_rows(starts, timings);
        const auto trace_lookup = index_starts(starts, offset);
        int total = 0;
        int matched = 0;
        int start_matched = 0;
        bind_stream(host, output, timings, starts, trace_lookup, total, matched, start_matched);
        std::cout << "nodes=" << total << " matched=" << matched
                  << " unmatched=" << total - matched
                  << " start_matched=" << start_matched << " sequence_offset=" << offset << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
