#include <atomic>
#include <climits>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

//--------- Slightly modified lazy version of SetList from the lecture --------

class Node {
public:
    std::mutex lock;
    std::string item;
    unsigned long key;
    std::shared_ptr<Node> next;
    bool marked;
    Node(const std::string& s, std::shared_ptr<Node> n = NULL) {
        item = s;
        key = std::hash<std::string>{}(s);
        next = n;
        marked = false;
    }
    Node(unsigned long k) {
        item = "";
        key = k;
        next = NULL; 
        marked = false;
    }
};

class SetList{
    static const unsigned long LOWEST_KEY;
    static const unsigned long LARGEST_KEY;
    std::shared_ptr<Node> head;

    static bool validate(const std::shared_ptr<const Node> pred, const std::shared_ptr<const Node> curr);
public:
    SetList() { // constructor
        head = std::make_shared<Node>(SetList::LOWEST_KEY);
        head->next = std::make_shared<Node>(SetList::LARGEST_KEY);
    }
    ~SetList() {}; // destructor
    bool add(const std::string& val);
    bool remove(const std::string& val);
    bool contains(const std::string& val) const;
};

const unsigned long SetList::LOWEST_KEY = 0;
const unsigned long SetList::LARGEST_KEY = ULONG_MAX;
 
bool SetList::validate(std::shared_ptr<const Node> pred, std::shared_ptr<const Node> curr) {
  return (!pred->marked) && (!curr->marked) && (pred->next == curr);
}

bool SetList::add(const std::string& val) {
    unsigned long key = std::hash<std::string>{}(val);
    while (true) {
        std::shared_ptr<Node> pred = this->head;
        std::shared_ptr<Node> curr = pred->next;
        while (curr->key < key) {
            pred = curr;
            curr = curr->next;
        }
        std::lock_guard<std::mutex> pred_lk(pred->lock);
        std::lock_guard<std::mutex> curr_lk(curr->lock);
        if (SetList::validate(pred, curr)) {
            if (key == curr->key) {
                return false;
            }
            pred->next = std::make_shared<Node>(val, curr);
            return true;
        }
    }
}
    
bool SetList::remove(const std::string& val) {
    unsigned long key = std::hash<std::string>{}(val);
    while (true) {
        std::shared_ptr<Node> pred = this->head;
        std::shared_ptr<Node> curr = pred->next;
        while (curr->key < key) {
            pred = curr;
            curr = curr->next;
        }
        std::lock_guard<std::mutex> pred_lk(pred->lock);
        std::lock_guard<std::mutex> curr_lk(curr->lock);
        if (SetList::validate(pred, curr)) {
            if (key == curr->key) {
                curr->marked = true;
            	pred->next = curr->next;
	            return true;
            }
	        return false;
        }
    }
}

bool SetList::contains(const std::string& val) const {
    unsigned long key = std::hash<std::string>{}(val);
    std::shared_ptr<Node> curr = this->head;
    while(curr->key < key)
        curr = curr->next;
    return (curr->key == key) && (!curr->marked);
}

//-----------------------------------------------------------------------------

class MultiNode {
public:
    std::mutex lock;
    std::string item;
    unsigned long key;
    std::shared_ptr<MultiNode> next;
    unsigned int multiplicity;

    MultiNode(const std::string& s, std::shared_ptr<MultiNode> n = NULL) {
        item = s;
        key = std::hash<std::string>{}(s);
        next = n;
        multiplicity = 1;
    }
    MultiNode(unsigned long k) {
        item = "";
        key = k;
        next = NULL; 
        multiplicity = 1;
    }
};

class MultiSetList{
    static const unsigned long LOWEST_KEY;
    static const unsigned long LARGEST_KEY;
    std::shared_ptr<MultiNode> head;

    static bool validate(std::shared_ptr<const MultiNode> pred, std::shared_ptr<const MultiNode> curr);
public:
    MultiSetList() { // constructor
        head = std::make_shared<MultiNode>(MultiSetList::LOWEST_KEY);
        head->next = std::make_shared<MultiNode>(MultiSetList::LARGEST_KEY);
    }
    void add(const std::string& val);
    bool remove(const std::string& val);
    unsigned int contains(const std::string& val) const;
};

const unsigned long MultiSetList::LOWEST_KEY = 0;
const unsigned long MultiSetList::LARGEST_KEY = ULONG_MAX;
 
bool MultiSetList::validate(std::shared_ptr<const MultiNode> pred, std::shared_ptr<const MultiNode> curr) {
  return (pred->multiplicity > 0) && (curr->multiplicity > 0) && (pred->next == curr);
}

void MultiSetList::add(const std::string& val) {
    unsigned long key = std::hash<std::string>{}(val);
    while (true) {
        std::shared_ptr<MultiNode> pred = this->head;
        std::shared_ptr<MultiNode> curr = pred->next;
        while (curr->key < key) {
            pred = curr;
            curr = curr->next;
        }
        std::lock_guard<std::mutex> pred_lk(pred->lock);
        std::lock_guard<std::mutex> curr_lk(curr->lock);
        if (MultiSetList::validate(pred, curr)) {
            if (key == curr->key) {
                curr->multiplicity++;
                break; // no return statement so should break
            }
            else {
                std::shared_ptr<MultiNode> node = std::make_shared<MultiNode>(val, curr);
                pred->next = node;
                break; // same
            }
            
        }
    }
}
    
bool MultiSetList::remove(const std::string& val) {
    unsigned long key = std::hash<std::string>{}(val);
    while (true) {
        std::shared_ptr<MultiNode> pred = this->head;
        std::shared_ptr<MultiNode> curr = pred->next;
        while (curr->key < key) {
            pred = curr;
            curr = curr->next;
        }
        std::lock_guard<std::mutex> pred_lk(pred->lock);
        std::lock_guard<std::mutex> curr_lk(curr->lock);
        if (MultiSetList::validate(pred, curr)) {
            if ( key == curr->key) {
                if (curr->multiplicity == 1){ //removing the element will result in its complete removal from the list 
                    pred->next = curr->next;
                }
                else { // multiplicity >=2
                    curr->multiplicity--;
                } 
                return true;
            } 
            else {
                return false;
            }
        }
    }
}

unsigned int MultiSetList::contains(const std::string& val) const {
    unsigned long key = std::hash<std::string>{}(val);
    std::shared_ptr<MultiNode> curr = this->head;
    while (curr->key < key)
        curr = curr->next;
    if (curr->key == key) {
        return curr->multiplicity;
    }
    return 0;
}

//-----------------------------------------------------------------------------

class AtomicNode {
public:
    std::string item;
    unsigned long key;
    std::atomic<AtomicNode*> next;
    
    AtomicNode(const std::string& s, AtomicNode* n = NULL) {
        item = s;
        key = std::hash<std::string>{}(s);
        std::atomic_init<AtomicNode*>(&next, n);
    }
    AtomicNode(unsigned long k) {
        item = "";
        key = k;
        next = NULL; 
    }
};


class MonotonicLockFreeSet {
    AtomicNode* head;

    static const unsigned long LOWEST_KEY;
    static const unsigned long LARGEST_KEY;
public:
    MonotonicLockFreeSet() {
        this->head = new AtomicNode(MonotonicLockFreeSet::LOWEST_KEY);
        this->head->next = new AtomicNode(MonotonicLockFreeSet::LARGEST_KEY);
    }
    ~MonotonicLockFreeSet();
    bool add(const std::string& val);
    bool contains(const std::string& val) const;
};

const unsigned long MonotonicLockFreeSet::LOWEST_KEY = 0;
const unsigned long MonotonicLockFreeSet::LARGEST_KEY = ULONG_MAX;

bool MonotonicLockFreeSet::add(const std::string& val) {
    unsigned long key = std::hash<std::string>{}(val);
    while (true) {
        AtomicNode* pred = this->head;
        AtomicNode* curr = pred->next;
        while (curr->key < key) {
            pred = curr;
            curr = curr->next;
        }
        if (key == curr->key) {
            return false;
        }
        AtomicNode* node = new AtomicNode(val, curr);
        
        if (pred->next.compare_exchange_strong(curr, node)) {
            return true;
        }
    }
}

bool MonotonicLockFreeSet::contains(const std::string& val) const {
    unsigned long key = std::hash<std::string>{}(val);
    AtomicNode* curr = this->head;
    while (curr->key < key) {
        curr = curr->next;
    }
    return curr->key == key;
}

MonotonicLockFreeSet::~MonotonicLockFreeSet() {
    AtomicNode* pred = this->head;
    AtomicNode* curr = pred->next;
    while (curr != NULL) {
        delete pred;
        pred = curr;
        curr = curr->next;
    }
    delete pred;
}

//-----------------------------------------------------------------------------

template <typename T>
struct RefAndMark {
    T* ref;
    bool mark;
    RefAndMark(T* _ref, bool _mark) {
        this->ref = _ref;
        this->mark = _mark;
    }
};

template <typename T>
class AtomicMarkable {
    std::atomic<RefAndMark<T> > data;
public:
    AtomicMarkable(T* _ref = nullptr, bool mark = false): data(RefAndMark<T>(_ref, mark)) {}

    bool compare_and_set(T*& expected_ref, bool& expected_mark, T* desired_ref, bool desired_mark) {
        return false;
    }

    bool attempt_mark(T* expected_ref, bool new_mark) {
        return true;
    }

    T* get(bool& mark) const {
    }
};


