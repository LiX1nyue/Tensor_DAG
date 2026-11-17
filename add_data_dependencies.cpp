#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Small JSON reader. Original text is retained separately for verbatim output.
struct Json {
    char kind = 0;
    std::string text;
    std::vector<Json> array;
    std::map<std::string, Json> object;
    const Json& at(const std::string& key) const {
        static const Json empty;
        auto it = object.find(key);
        return it == object.end() ? empty : it->second;
    }
    int64_t integer() const { return std::stoll(text); }
};

struct Parser {
    const std::string& source;
    size_t pos = 0;
    void space() { while (pos < source.size() && std::isspace((unsigned char)source[pos])) ++pos; }
    char peek() { space(); return source.at(pos); }
    void take(char c) { space(); if (source.at(pos++) != c) throw std::runtime_error("Invalid JSON delimiter"); }
    std::string string() {
        take('"');
        std::string result;
        while (true) {
            char c = source.at(pos++);
            if (c == '"') return result;
            if (c != '\\') { result += c; continue; }
            c = source.at(pos++);
            if (c == 'n') result += '\n';
            else if (c == 'r') result += '\r';
            else if (c == 't') result += '\t';
            else if (c == 'b') result += '\b';
            else if (c == 'f') result += '\f';
            else if (c == 'u') {
                auto hex = [&]() { auto v = std::stoul(source.substr(pos, 4), nullptr, 16); pos += 4; return v; };
                unsigned long u = hex();
                if (u >= 0xd800 && u <= 0xdbff) {
                    if (source.substr(pos, 2) != "\\u") throw std::runtime_error("Invalid surrogate");
                    pos += 2; auto low = hex();
                    if (low < 0xdc00 || low > 0xdfff) throw std::runtime_error("Invalid surrogate");
                    u = 0x10000 + ((u - 0xd800) << 10) + low - 0xdc00;
                }
                if (u < 128) result += char(u);
                else if (u < 2048) { result += char(0xc0 | (u >> 6)); result += char(0x80 | (u & 63)); }
                else if (u < 65536) { result += char(0xe0 | (u >> 12)); result += char(0x80 | ((u >> 6) & 63)); result += char(0x80 | (u & 63)); }
                else { result += char(0xf0 | (u >> 18)); result += char(0x80 | ((u >> 12) & 63)); result += char(0x80 | ((u >> 6) & 63)); result += char(0x80 | (u & 63)); }
            } else if (c == '"' || c == '\\' || c == '/') result += c;
            else throw std::runtime_error("Invalid JSON escape");
        }
    }
    Json value() {
        Json j; j.kind = peek();
        if (j.kind == '"') j.text = string();
        else if (j.kind == '[') {
            take('[');
            if (peek() != ']') do { j.array.push_back(value()); if (peek() != ',') break; take(','); } while (true);
            take(']');
        } else if (j.kind == '{') {
            take('{');
            if (peek() != '}') do { auto k = string(); take(':'); j.object.emplace(k, value()); if (peek() != ',') break; take(','); } while (true);
            take('}');
        } else {
            size_t start = pos;
            while (pos < source.size() && !std::isspace((unsigned char)source[pos]) && source[pos] != ',' && source[pos] != ']' && source[pos] != '}') ++pos;
            if (pos == start) throw std::runtime_error("Invalid JSON value");
            j.text = source.substr(start, pos - start);
        }
        return j;
    }
};

struct Tensor {
    std::string device;
    int64_t storage, offset, size, width = 1;
    std::vector<std::pair<int64_t, int64_t>> outer;
};
const Json& child(const Json& j, size_t i) {
    static const Json empty;
    return i < j.array.size() ? j.array[i] : empty;
}
void tensors(const Json& values, const Json& shapes, const Json& strides, std::vector<Tensor>& out) {
    if (values.kind != '[') return;
    const auto& a = values.array;
    bool tensor = a.size() == 6 && a[5].kind == '"';
    for (size_t i = 0; tensor && i < 5; ++i)
        tensor = a[i].kind == '-' || (a[i].kind >= '0' && a[i].kind <= '9');
    if (!tensor) {
        for (size_t i = 0; i < a.size(); ++i) tensors(a[i], child(shapes, i), child(strides, i), out);
        return;
    }
    Tensor t; t.device = a[5].text; t.storage = a[1].integer(); t.offset = a[2].integer(); t.size = a[4].integer();
    if (!t.storage || a[3].integer() == 0) return;
    if (t.size <= 0 || t.offset < 0) throw std::runtime_error("Invalid tensor storage metadata");
    if (shapes.array.size() != strides.array.size() || (shapes.array.empty() && a[3].integer() != 1))
        throw std::runtime_error("Missing tensor shape/stride");
    std::vector<std::pair<int64_t, int64_t>> dims;
    for (size_t i = 0; i < shapes.array.size(); ++i) {
        auto count = shapes.array[i].integer(), stride = strides.array[i].integer();
        if (!count) return;
        if (count < 0 || stride < 0) throw std::runtime_error("Negative shape/stride unsupported");
        if (count > 1 && stride > 0) dims.emplace_back(stride, count);
    }
    std::sort(dims.begin(), dims.end());
    for (auto [stride, count] : dims) {
        if (t.outer.empty() && stride <= t.width) t.width += stride * (count - 1);
        else t.outer.emplace_back(stride, count);
    }
    out.push_back(std::move(t));
}
void runs(const Tensor& t, const std::function<void(int64_t,int64_t)>& visit, size_t dim = 0, int64_t delta = 0) {
    if (dim == t.outer.size()) { visit((t.offset + delta) * t.size, (t.offset + delta + t.width) * t.size); return; }
    auto [stride, count] = t.outer[dim];
    for (int64_t i = 0; i < count; ++i) runs(t, visit, dim + 1, delta + i * stride);
}

struct Node {
    int64_t id;
    long double time = 0;
    bool timed = false;
    size_t begin, end;
    std::vector<Tensor> inputs, outputs;
    std::set<int64_t> deps;
};
struct Span { int64_t end; size_t generation; std::set<int64_t> owners; };
class Writers {
    std::map<int64_t, Span> spans;
    auto split(int64_t x) {
        auto it = spans.lower_bound(x);
        if (it != spans.end() && it->first == x) return it;
        if (it != spans.begin()) {
            auto prev = std::prev(it);
            if (prev->second.end > x) {
                auto tail = prev->second; prev->second.end = x;
                return spans.emplace_hint(it, x, std::move(tail));
            }
        }
        return it;
    }
public:
    void read(int64_t left, int64_t right, std::set<int64_t>& deps) const {
        auto it = spans.upper_bound(left);
        if (it != spans.begin()) --it;
        for (; it != spans.end() && it->first < right; ++it)
            if (it->second.end > left) deps.insert(it->second.owners.begin(), it->second.owners.end());
    }
    void write(int64_t left, int64_t right, size_t generation, int64_t id) {
        auto last = split(right), it = split(left);
        int64_t cursor = left;
        while (it != last) {
            if (cursor < it->first) spans.emplace_hint(it, cursor, Span{it->first, generation, {id}});
            cursor = it->second.end;
            if (it->second.generation != generation) it->second.owners.clear();
            it->second.generation = generation; it->second.owners.insert(id); ++it;
        }
        if (cursor < right) spans.emplace_hint(last, cursor, Span{right, generation, {id}});
        // Merge adjacent equal spans, including the two boundary neighbours.
        it = spans.lower_bound(left);
        if (it != spans.begin()) --it;
        while (it != spans.end()) {
            auto next = std::next(it);
            if (next == spans.end() || next->first > right) break;
            if (it->second.end == next->first && it->second.generation == next->second.generation && it->second.owners == next->second.owners) {
                it->second.end = next->second.end; spans.erase(next);
            } else it = next;
        }
    }
};

int main(int argc, char** argv) {
    try {
        std::string input, output;
        for (int i = 1; i < argc; ++i) {
            std::string option = argv[i];
            if (i + 1 == argc) throw std::runtime_error("Missing argument value");
            if (option == "--input") input = argv[++i];
            else if (option == "--output") output = argv[++i];
            else throw std::runtime_error("Unknown option: " + option);
        }
        if (input.empty() || output.empty()) throw std::runtime_error("Usage: add_data_dependencies --input BOUND.json --output DAG.json");
        if (std::filesystem::weakly_canonical(input) == std::filesystem::weakly_canonical(output)) throw std::runtime_error("Use a distinct output file");
        std::ifstream file(input, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open input");
        std::string source((std::istreambuf_iterator<char>(file)), {});
        Parser p{source}; p.take('{'); bool found = false;
        std::vector<Node> nodes;
        while (p.peek() != '}') {
            auto key = p.string(); p.take(':');
            if (key == "nodes") {
                found = true; p.take('[');
                while (p.peek() != ']') {
                    Node n; n.begin = p.pos; auto j = p.value(); n.end = p.pos - 1;
                    if (j.object.count("data_deps")) throw std::runtime_error("Input already has data_deps");
                    n.id = j.at("id").integer();
                    const auto& ts = j.at("start_ts");
                    if (ts.kind && ts.text != "null") { n.time = std::stold(ts.text); n.timed = std::isfinite(n.time); }
                    for (auto side : {"inputs", "outputs"}) {
                        const auto& io = j.at(side);
                        tensors(io.at("values"), io.at("shapes"), io.at("strides"), std::string(side) == "inputs" ? n.inputs : n.outputs);
                    }
                    nodes.push_back(std::move(n));
                    if (p.peek() != ',') break;
                    p.take(',');
                }
                p.take(']');
            } else p.value();
            if (p.peek() != ',') break;
            p.take(',');
        }
        p.take('}');
        if (!found) throw std::runtime_error("Missing nodes array");
        std::vector<size_t> order;
        for (size_t i = 0; i < nodes.size(); ++i) if (nodes[i].timed) order.push_back(i);
        if (order.empty()) throw std::runtime_error("No per-node start_ts; bind timestamps first");
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return nodes[a].time < nodes[b].time; });
        std::map<std::pair<std::string,int64_t>, Writers> writers;
        for (size_t begin = 0; begin < order.size();) {
            size_t end = begin + 1;
            while (end < order.size() && nodes[order[end]].time == nodes[order[begin]].time) ++end;
            for (size_t k = begin; k < end; ++k) {
                auto& n = nodes[order[k]];
                for (const auto& t : n.inputs) runs(t, [&](auto l, auto r) { writers[{t.device,t.storage}].read(l,r,n.deps); });
            }
            for (size_t k = begin; k < end; ++k) {
                auto& n = nodes[order[k]];
                for (const auto& t : n.outputs) runs(t, [&](auto l, auto r) { writers[{t.device,t.storage}].write(l,r,begin,n.id); });
            }
            begin = end;
        }
        auto parent = std::filesystem::path(output).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream out(output, std::ios::binary);
        if (!out) throw std::runtime_error("Cannot open output");
        size_t cursor = 0, edges = 0;
        for (const auto& n : nodes) {
            size_t insertion = n.end;
            while (insertion > n.begin && std::isspace((unsigned char)source[insertion-1])) --insertion;
            out.write(source.data()+cursor, insertion-cursor);
            auto ws = source.substr(insertion, n.end-insertion);
            if (ws.find('\n') != std::string::npos) {
                size_t first = n.begin+1;
                while (std::isspace((unsigned char)source[first])) ++first;
                size_t line = source.rfind('\n', first);
                std::string indent = line != std::string::npos && line >= n.begin ? source.substr(line+1,first-line-1) : "  ";
                out << ',' << (ws.find("\r\n") != std::string::npos ? "\r\n" : "\n") << indent;
            } else out << ", ";
            out << "\"data_deps\": ["; bool first = true;
            for (auto id : n.deps) { if (!first) out << ", "; out << id; first = false; }
            out << ']'; cursor = insertion; edges += n.deps.size();
        }
        out.write(source.data()+cursor, source.size()-cursor); out.close();
        if (!out) throw std::runtime_error("Output write failed");
        std::cout << "nodes=" << nodes.size() << " timed=" << order.size() << " missing_time=" << nodes.size()-order.size() << " edges=" << edges << '\n';
    } catch (const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
