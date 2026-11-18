#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
#include <iterator>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Json {
    char kind = 0;
    std::string text;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    const Json* get(const std::string& key) const {
        auto it = object.find(key);
        return it == object.end() ? nullptr : &it->second;
    }
    std::string str() const { return text; }
    int64_t integer(int64_t fallback = -1) const {
        try { return std::stoll(text); } catch (...) { return fallback; }
    }
};

struct Parser {
    const std::string& text;
    size_t pos = 0;
    void ws() { while (pos < text.size() && std::isspace((unsigned char)text[pos])) ++pos; }
    char peek() { ws(); if (pos >= text.size()) throw std::runtime_error("Unexpected end of JSON"); return text[pos]; }
    void expect(char c) { ws(); if (pos >= text.size() || text[pos++] != c) throw std::runtime_error("Malformed JSON"); }
    std::string string() {
        expect('"');
        std::string out;
        while (pos < text.size()) {
            char c = text[pos++];
            if (c == '"') return out;
            if (c != '\\') { out += c; continue; }
            if (pos >= text.size()) break;
            c = text[pos++];
            switch (c) {
                case '"': case '\\': case '/': out += c; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    if (pos + 4 > text.size()) throw std::runtime_error("Invalid JSON unicode escape");
                    unsigned value = std::stoul(text.substr(pos, 4), nullptr, 16); pos += 4;
                    if (value < 0x80) out += char(value);
                    else if (value < 0x800) { out += char(0xc0 | value >> 6); out += char(0x80 | (value & 63)); }
                    else { out += char(0xe0 | value >> 12); out += char(0x80 | ((value >> 6) & 63)); out += char(0x80 | (value & 63)); }
                    break;
                }
                default: throw std::runtime_error("Invalid JSON escape");
            }
        }
        throw std::runtime_error("Unterminated JSON string");
    }
    Json value() {
        Json result; result.kind = peek();
        if (result.kind == '"') result.text = string();
        else if (result.kind == '{') {
            expect('{');
            if (peek() != '}') while (true) {
                auto key = string(); expect(':'); result.object.emplace(key, value());
                if (peek() != ',') break;
                expect(',');
            }
            expect('}');
        } else if (result.kind == '[') {
            expect('[');
            if (peek() != ']') while (true) {
                result.array.push_back(value());
                if (peek() != ',') break;
                expect(',');
            }
            expect(']');
        } else {
            size_t begin = pos;
            while (pos < text.size() && !std::isspace((unsigned char)text[pos]) &&
                   text[pos] != ',' && text[pos] != ']' && text[pos] != '}') ++pos;
            result.text = text.substr(begin, pos - begin);
        }
        return result;
    }
};

struct Group {
    std::vector<int> ranks;
    int size = 0;
    std::string description;
};
struct Comm {
    int rank;
    int64_t id;
    std::string name;
    std::string type;
    std::string group;
    std::string group_key;
    std::string members;
    long double ts = 0;
    std::string duration;
    std::vector<int64_t> children;
    int64_t seq = -1;
    std::string inputs;
    std::string outputs;
    std::string root_arg;
    std::vector<std::string> candidate_groups;
    std::string p2p_peer;
};

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open " + path);
    return std::string(std::istreambuf_iterator<char>(in), {});
}

std::string compact(const Json& value) {
    if (value.kind == '"') {
        std::ostringstream out; out << '"';
        for (unsigned char c : value.text) {
            if (c == '"' || c == '\\') out << '\\' << char(c);
            else if (c < 0x20) out << "\\u00" << "0123456789abcdef"[c >> 4] << "0123456789abcdef"[c & 15];
            else out << char(c);
        }
        out << '"'; return out.str();
    }
    if (value.kind == '[') {
        std::string out = "[";
        for (size_t i=0;i<value.array.size();++i) { if (i) out += ','; out += compact(value.array[i]); }
        return out + ']';
    }
    if (value.kind == '{') {
        std::string out = "{"; bool first = true;
        for (const auto& [key,val] : value.object) {
            if (!first) out += ',';
            first = false;
            Json k; k.kind='"'; k.text=key; out += compact(k) + ':' + compact(val);
        }
        return out + '}';
    }
    return value.text;
}

std::vector<int> parse_ranks(const Json& group) {
    std::vector<int> ranks;
    if (const auto* values = group.get("ranks"))
        for (const auto& value : values->array) ranks.push_back((int)value.integer());
    if (ranks.empty() && group.get("group_size")) {
        for (int i=0;i<group.get("group_size")->integer();++i) ranks.push_back(i);
    }
    std::sort(ranks.begin(), ranks.end());
    return ranks;
}

std::string group_key(const std::vector<int>& ranks) {
    std::ostringstream out;
    for (size_t i=0;i<ranks.size();++i) { if (i) out << ','; out << ranks[i]; }
    return out.str();
}

std::string comm_type(const std::string& name) {
    if (name == "c10d::allreduce_") return "allreduce";
    if (name == "c10d::broadcast_") return "broadcast";
    if (name == "c10d::_allgather_base_") return "allgather";
    if (name == "c10d::_reduce_scatter_base_") return "reducescatter";
    if (name == "c10d::barrier") return "barrier";
    if (name == "c10d::send") return "send";
    if (name == "c10d::recv_") return "recv";
    return {};
}

bool compatible_type(const std::string& group_name, const std::string& type) {
    if (group_name.find("DATA_PARALLEL") != std::string::npos ||
        group_name.find("TENSOR_AND_DATA_PARALLEL") != std::string::npos ||
        group_name.find("EXPERT_DATA_PARALLEL") != std::string::npos)
        return type == "allreduce" || type == "allgather" || type == "reducescatter" || type == "broadcast" || type == "barrier";
    if (group_name.find("MODEL_PARALLEL") != std::string::npos ||
        group_name.find("PIPELINE_MODEL_PARALLEL") != std::string::npos ||
        group_name.find("EMBEDDING") != std::string::npos)
        return type == "send" || type == "recv" || type == "broadcast" || type == "barrier";
    return type == "barrier";
}

std::vector<std::string> candidate_groups(const Comm& c, const std::map<std::string, Group>& groups) {
    std::vector<std::string> compatible;
    for (const auto& [key, g] : groups) {
        if (key.rfind(std::to_string(c.rank)+":",0)!=0 || g.ranks.empty()) continue;
        if (std::find(g.ranks.begin(),g.ranks.end(),c.rank)==g.ranks.end()) continue;
        if (g.ranks.size() > 1 && compatible_type(g.description, c.type))
            compatible.push_back(g.description + "|" + group_key(g.ranks));
    }
    std::sort(compatible.begin(),compatible.end());
    compatible.erase(std::unique(compatible.begin(),compatible.end()),compatible.end());
    return compatible;
}

bool is_comm_root(const std::string& name) { return !comm_type(name).empty(); }

std::string attr_string(const Json& node, const std::string& key) {
    const auto* attrs = node.get("attrs");
    if (!attrs) return {};
    for (const auto& a : attrs->array) {
        const auto* name = a.get("name");
        const auto* value = a.get("value");
        if (name && value && name->str() == key) return value->str();
    }
    return {};
}

std::vector<Comm> read_rank(int rank, const std::string& path, std::map<std::string, Group>& all_groups) {
    const auto text = read_file(path);
    Parser parser{text};
    const Json root = parser.value();
    const auto* nodes_json = root.get("nodes");
    if (!nodes_json) throw std::runtime_error("No nodes array in " + path);
    std::unordered_map<int64_t, const Json*> by_id;
    std::unordered_map<int64_t, std::vector<const Json*>> children;
    for (const auto& n : nodes_json->array) {
        const auto id = n.get("id")->integer(); by_id[id] = &n;
        if (const auto* parent = n.get("ctrl_deps")) children[parent->integer()].push_back(&n);
        if (n.get("name") && n.get("name")->str() == "## process_group:init ##") {
            const auto* vals = n.get("inputs")->get("values");
            if (vals && !vals->array.empty() && vals->array[0].kind == '"') {
                Parser gp{vals->array[0].text}; const Json groups = gp.value();
                for (const auto& g : groups.array) {
                    const auto* pg = g.get("pg_name");
                    if (!pg) continue;
                    Group info; info.ranks = parse_ranks(g);
                    info.size = g.get("group_size") ? (int)g.get("group_size")->integer() : (int)info.ranks.size();
                    if (g.get("pg_desc")) info.description = g.get("pg_desc")->str();
                    if (g.get("pg_desc")) info.description = g.get("pg_desc")->str();
                    all_groups[std::to_string(rank) + ":" + pg->str()] = info;
                }
            }
        }
    }
    // Resolve process-group metadata after collecting this rank's groups.
    // Keep all compatible member sets: identical rank lists may represent
    // distinct process groups and therefore remain ambiguous.
    std::vector<Comm> result;
    for (const auto& n : nodes_json->array) {
        if (!n.get("name") || !is_comm_root(n.get("name")->str())) continue;
        Comm c{}; c.rank=rank; c.id=n.get("id")->integer(); c.name=n.get("name")->str(); c.type=comm_type(c.name);
        c.ts=0;
        if (const auto* ts=n.get("start_ts")) { try { c.ts=std::stold(ts->text); } catch (...) {} }
        if (const auto* d=n.get("duration")) c.duration=d->text;
        try { c.seq=std::stoll(attr_string(n,"seq_id")); } catch (...) {}
        if (const auto* v=n.get("inputs")) c.inputs=compact(*v);
        if (const auto* v=n.get("outputs")) c.outputs=compact(*v);
        if (c.type=="broadcast" && n.get("inputs") && n.get("inputs")->get("values")) {
            const auto& vals=n.get("inputs")->get("values")->array;
            if (vals.size()>2) c.root_arg=vals[2].text;
        }

        // Associate the root with its HCCL descendant; preserve the highest
        // HCCL node as a local source ID, avoiding duplicate HCCL wrapper nodes.
        std::vector<int64_t> stack{c.id};
        std::set<int64_t> visited;
        std::vector<int64_t> hccl_ids;
        while (!stack.empty()) {
            auto current=stack.back(); stack.pop_back();
            if (!visited.insert(current).second) continue;
            auto it=by_id.find(current); if (it==by_id.end()) continue;
            const auto* node=it->second;
            if (node->get("name") && node->get("name")->str().rfind("Hccl",0)==0) hccl_ids.push_back(current);
            for (auto child : children[current]) stack.push_back(child->get("id")->integer());
        }
        std::sort(hccl_ids.begin(),hccl_ids.end()); c.children=std::move(hccl_ids);

        // Available generic-group evidence from host metadata: match group by
        // collective type and this rank's group membership. If several groups
        // qualify, keep it unresolved instead of guessing.
        c.candidate_groups=candidate_groups(c,all_groups);
        if(c.type=="send" || c.type=="recv") {
            const auto* vals=n.get("inputs") ? n.get("inputs")->get("values") : nullptr;
            if(vals && vals->array.size()>2) {
                const int peer=(int)vals->array[2].integer(-1);
                if(peer>=0) {
                    c.p2p_peer=std::to_string(peer);
                    c.group_key=c.type=="send" ? std::to_string(rank)+"->"+c.p2p_peer
                                                  : c.p2p_peer+"->"+std::to_string(rank);
                    c.members=c.group_key;
                }
            }
        }
        result.push_back(std::move(c));
    }
    return result;
}

std::string escape_json(const std::string& s) {
    std::ostringstream out; out << '"';
    for (unsigned char c:s) {
        switch(c) { case '"':out<<"\\\"";break;case '\\':out<<"\\\\";break;case '\n':out<<"\\n";break;case '\r':out<<"\\r";break;case '\t':out<<"\\t";break;default: if(c<0x20) out<<"?"; else out<<char(c); }
    }
    return out.str()+'"';
}

} // namespace

int main(int argc,char** argv) {
    try {
        std::string input_dir="output", output="output/host.8rank.bound.json", mapping;
        bool infer_dp = false;
        for(int i=1;i<argc;++i) {
            std::string option=argv[i];
            if(option=="--infer-dp") { infer_dp=true; continue; }
            if(i+1>=argc) throw std::runtime_error("Missing option value");
            if(option=="--input-dir") input_dir=argv[++i]; else if(option=="--output") output=argv[++i];
            else if(option=="--mapping") mapping=argv[++i];
            else throw std::runtime_error("Unknown option: "+option);
        }
        std::map<std::string,Group> groups;
        std::vector<std::vector<Comm>> per_rank(8);
        for(int r=0;r<8;++r) {
            const auto path=(std::filesystem::path(input_dir)/("host."+std::to_string(r)+".bound.json")).string();
            per_rank[r]=read_rank(r,path,groups);
        }
        std::map<std::string,std::set<int>> group_participants;
        for (const auto& [rank_group, group] : groups) {
            (void)rank_group;
            if (group.ranks.size() > 1)
                group_participants[group.description + "|" + group_key(group.ranks)] =
                    std::set<int>(group.ranks.begin(), group.ranks.end());
        }
        std::map<std::pair<int,int64_t>,Comm*> roots;
        for(auto& rank:per_rank) for(auto& c:rank) roots[{c.rank,c.id}]=&c;
        std::vector<std::vector<Comm*>> merged;
        std::set<Comm*> explicitly_mapped;
        std::set<std::pair<int,int64_t>> assigned;
        for(auto& rank:per_rank) for(auto& c:rank) {
            c.group_key.clear();
            if(!infer_dp) continue;
            if(c.type=="send" || c.type=="recv" || c.type=="broadcast" ||
               c.type=="allreduce" || c.type=="barrier") continue;
            if(c.candidate_groups.size()==1) c.group_key=c.candidate_groups.front();
            else if(c.candidate_groups.size()>1) {
                std::vector<std::string> filtered;
                for(const auto& candidate:c.candidate_groups) {
                    const auto sep=candidate.find('|');
                    const auto desc=sep==std::string::npos?candidate:candidate.substr(0,sep);
                    if((c.type=="allgather" || c.type=="reducescatter") &&
                       (desc=="DATA_PARALLEL_GROUP" ||
                        desc=="TENSOR_AND_DATA_PARALLEL_GROUP" ||
                        desc=="DATA_PARALLEL_GROUP_WITH_CP" ||
                        desc=="TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP" ||
                        desc=="EXPERT_DATA_PARALLEL_GROUP"))
                        filtered.push_back(candidate);
                }
                for (const auto& candidate : filtered) {
                    const auto split=candidate.find('|');
                    if (split==std::string::npos) continue;
                    const auto members=candidate.substr(split+1);
                    auto prev=c.group_key;
                    if(prev.empty()) c.group_key=candidate;
                    else if(prev.substr(prev.find('|')+1)!=members) { c.group_key.clear(); break; }
                    else if(candidate.find("TENSOR_AND_DATA_PARALLEL_GROUP")!=std::string::npos ||
                            candidate.find("DATA_PARALLEL_GROUP_WITH_CP")!=std::string::npos ||
                            candidate.find("EXPERT_DATA_PARALLEL_GROUP")!=std::string::npos) c.group_key=candidate;
                }
            }
        }
        std::map<std::pair<int,std::string>,size_t> ordinals;
        std::map<std::tuple<std::string,std::string,size_t>,std::vector<Comm*>> inferred;
        for(auto& rank:per_rank) for(auto& c:rank) {
            if(c.group_key.empty()) continue;
            const auto signature=c.type+":"+c.root_arg;
            const auto ordinal_key=std::make_pair(c.rank,c.group_key+":"+signature);
            const auto index=ordinals[ordinal_key]++;
            inferred[{c.group_key,signature,index}].push_back(&c);
        }
        for(auto& [key,members]:inferred) {
            const auto& group=std::get<0>(key);
            if(members.size()<2) continue;
            std::set<int> present;
            for(const auto* c:members) present.insert(c->rank);
            auto expected=group_participants.find(group);
            if(expected==group_participants.end() || present!=expected->second) continue;
            for(const auto* c:members) assigned.insert({c->rank,c->id});
            merged.push_back(std::move(members));
        }
        if(!mapping.empty()) {
            auto text=read_file(mapping); Parser parser{text}; auto list=parser.value();
            if(list.kind!='[') throw std::runtime_error("Mapping must be an array of member arrays");
            for(const auto& instance:list.array) {
                if(instance.kind!='[' || instance.array.size()<2) throw std::runtime_error("Mapping instance needs at least two ranks");
                std::vector<Comm*> members; std::set<int> ranks; std::string type;
                for(const auto& member:instance.array) {
                    if(!member.get("rank") || !member.get("node_id")) throw std::runtime_error("Missing mapping rank/node_id");
                    auto key=std::make_pair((int)member.get("rank")->integer(),member.get("node_id")->integer());
                    if(!roots.count(key) || assigned.count(key) || !ranks.insert(key.first).second)
                        throw std::runtime_error("Unknown or duplicated mapping member");
                    auto* c=roots.at(key);
                    if(type.empty()) type=c->type;
                    if(type!=c->type) throw std::runtime_error("Mapping operation types differ");
                    if(type=="send" || type=="recv") throw std::runtime_error("P2P merging requires peer/context evidence; unsupported mapping");
                    members.push_back(c);
                }
                for (auto* member : members) {
                    assigned.insert({member->rank, member->id});
                    explicitly_mapped.insert(member);
                }
                merged.push_back(std::move(members));
            }
        }
        for(auto& rank:per_rank) for(auto& c:rank)
            if(!assigned.count({c.rank,c.id})) merged.push_back({&c});

        std::ostringstream extra;
        extra << std::setprecision(std::numeric_limits<long double>::max_digits10);
        size_t global_count=0, cross_rank_count=0;
        std::map<std::pair<int,int64_t>,std::string> aliases;
        for(const auto& members:merged) {
            const auto type=members.front()->type;
            if(members.size()>1) ++cross_rank_count;
            auto global_id="comm:"+std::to_string(global_count);
            extra << "    {\"id\": " << escape_json(global_id)
                  << ", \"name\": \"GLOBAL_COMM_" << type.substr(0,type.find(':')) << "\", \"comm_type\": " << escape_json(type.substr(0,type.find(':')))
                  << ", \"match_status\": " << escape_json(members.size()>1 ?
                      (explicitly_mapped.count(members.front()) ? "explicit_mapping" : "dp_membership_and_order_hypothesis") : "rank_local_unmerged")
                  << ", \"group_hypothesis\": " << escape_json(members.front()->group_key)
                  << ", \"members\": [";
            bool first=true;
            for(const auto* c:members) {
                if(!first) extra << ',';
                first=false;
                aliases[{c->rank,c->id}]=global_id;
                for(auto id:c->children) aliases[{c->rank,id}]=global_id;
                extra << "{\"rank\":" << c->rank << ",\"node_id\":" << c->id
                      << ",\"start_ts\":" << c->ts << ",\"duration\":" << (c->duration.empty()?"null":c->duration)
                      << ",\"local_op\":" << escape_json(c->name)
                      << ",\"hccl_node_ids\":[";
                for(size_t j=0;j<c->children.size();++j) { if(j) extra << ','; extra << c->children[j]; }
                extra << "]}";
            }
            extra << "]},\n"; ++global_count;
        }

        if(!std::filesystem::path(output).parent_path().empty()) std::filesystem::create_directories(std::filesystem::path(output).parent_path());
        for(int r=0;r<8;++r) if(std::filesystem::weakly_canonical(output)==std::filesystem::weakly_canonical(std::filesystem::path(input_dir)/("host."+std::to_string(r)+".bound.json"))) throw std::runtime_error("Output cannot overwrite input");
        std::ofstream out(output,std::ios::binary);
        if(!out) throw std::runtime_error("Cannot open output file");
        out << "{\n  \"schema\": \"tensor.communication.merge.v1\",\n  \"source_rank_count\": 8,\n  \"communication_nodes\": [\n";
        auto generated=extra.str();
        if(!generated.empty()) generated.resize(generated.size()-2);
        out << generated << "\n  ],\n  \"ranks\": [\n";
        for(int r=0;r<8;++r) {
            if(r) out << ",\n";
            auto source=read_file((std::filesystem::path(input_dir)/("host."+std::to_string(r)+".bound.json")).string());
            Parser p{source}; p.expect('{'); bool found=false;
            while(p.peek()!='}') { auto k=p.string();p.expect(':');if(k=="nodes"){found=true;break;}p.value();if(p.peek()!=',')break;p.expect(','); }
            if(!found) throw std::runtime_error("No nodes array");
            p.expect('['); out << "    {\"rank\": " << r << ", \"nodes\": [\n";bool first=true;
            while(p.peek()!=']') {
                size_t begin=p.pos;auto n=p.value();auto id=n.get("id")->integer();
                if(!first) out << ",\n";
                first=false;
                auto alias=aliases.find({r,id});
                if(alias!=aliases.end()) {
                    out << "{\"id\":" << id << ",\"communication_ref\":" << escape_json(alias->second)
                        << ",\"source_node\":";out.write(source.data()+begin,p.pos-begin);out << '}';
                } else out.write(source.data()+begin,p.pos-begin);
                if(p.peek()!=',')break;
                p.expect(',');
            }
            p.expect(']');out << "\n]}";
        }
        out << "\n  ]\n}\n";
        if(!out) throw std::runtime_error("Failed writing output");
        size_t local=0; for(const auto& r:per_rank)local+=r.size();
        std::cout << "local_communication_roots=" << local << " merged_global_nodes=" << global_count
                  << " cross_rank_merged=" << cross_rank_count << " output=" << output << '\n';
    } catch(const std::exception& e) { std::cerr << "error: " << e.what() << '\n'; return 1; }
}
