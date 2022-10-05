#ifndef TYPES_H
#define TYPES_H
#include <algorithm>
#include <cstddef>
#include <list>
#include <string>
#include <utility>
#include <variant>
#include <cassert>

using Locus = size_t;
using Event = std::pair<Locus, Locus>;

std::pair<Event, Event> swap_breakpoints(Event brkp1, Event brkp2, int left,
                                         int right);

class SNVEvent {
public:
size_t snv; 
int lhs_locus; // -1 means there is none on the left hand side 

bool overlaps_with_event(Event e) {
  return e.first <= lhs_locus && e.second > lhs_locus;
}

bool operator==(const SNVEvent& a) const
    {
        return this->snv == a.snv;
    }

SNVEvent(size_t i, int l) {
  snv = i; 
  lhs_locus = l;
}
};

using TreeLabel = std::variant<Event, SNVEvent>;

inline bool is_cn_event(TreeLabel l) {return l.index() == 0;}

inline Locus get_event_start_locus(Event event) { return event.first; }

inline Locus get_event_end_locus(Event event) { return event.second; }

inline std::list<Locus> get_event_breakpoints(Event event) {
  return std::list<Locus>{event.first, event.second};
}

inline Event get_event_from_label(TreeLabel label) { 
  assert(is_cn_event(label));
  return std::get<0>(label); 
}

inline TreeLabel get_root_label() { return TreeLabel(std::make_pair(0, 0)); }

inline std::string label_to_str(TreeLabel label) {
  if (is_cn_event(label)) {
    auto label_ = std::get<0>(label);
    return "(" + std::to_string(label_.first) + "," +
         std::to_string(label_.second) + ")";
  } else {
    auto label_ = std::get<1>(label);
    return "(SNV_" + std::to_string(label_.snv) + ")";
  }
}

inline bool is_valid_event(const Event brkp) {
  return brkp.first < brkp.second;
}

inline bool is_root_event(const Event brkp) {
  return brkp.first == 0 && brkp.second == 0;
}

namespace std
{
    template<> struct less<TreeLabel>
    {
       bool operator() (const TreeLabel& lhs, const TreeLabel& rhs) const
       {  
            if (is_cn_event(lhs)) {
              if (is_cn_event(rhs)) {
                auto le = get_event_from_label(lhs);
                auto re = get_event_from_label(rhs);
                return le.first < re.first || (le.first == re.first && le.second < re.second);
              }
              return true;
            } else if (is_cn_event(rhs)) {
              return false;
            }
            return std::get<1>(lhs).snv < std::get<1>(rhs).snv; 
       }
    };
}


#endif // !TYPES_H
