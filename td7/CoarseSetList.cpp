class CoarseNode {
public:
    std::string item;
    unsigned long key;
    CoarseNode * next;
    CoarseNode() {}
    CoarseNode(const std::string& s) {
        this->item = s;
        this->key = std::hash<std::string>{}(s);
        this->next = NULL;
    }
    CoarseNode(unsigned long k) {
        this->item = "";
        this->key = k;
        this->next = NULL;
    }
};

void DeleteNodeChain(CoarseNode* start) {
    CoarseNode* prev = start;
    CoarseNode* cur = prev->next;
    while (cur != NULL) {
        delete prev;
        prev = cur;
        cur = cur->next;
    }
    delete prev;
}

class CoarseSetList {
protected:
    std::recursive_mutex lock;
    CoarseNode* head;
    static const unsigned long LOWEST_KEY = 0;
    static const unsigned long LARGEST_KEY = ULONG_MAX;
    // returns the pointer to the last node with key < hash(val)
    // with keeping this and the next nodes locked
    CoarseNode* search(const std::string& val);
public:
    CoarseSetList() {
        this->head = new CoarseNode(CoarseSetList::LOWEST_KEY);
        this->head->next = new CoarseNode(CoarseSetList::LARGEST_KEY);
    }
    ~CoarseSetList();
    bool add(const std::string& val);
    bool remove(const std::string& val);
    bool contains(const std::string& val);
    void print() const; // for testing
    unsigned long size() const; // for testing
};

unsigned long CoarseSetList::size() const {
    unsigned long result = 0;
    CoarseNode* cur = this->head->next;
    while (cur->next != NULL) {
        ++result;
        cur = cur->next;
    }
    return result;
}

CoarseSetList::~CoarseSetList() {
    DeleteNodeChain(this->head);
}

CoarseNode* CoarseSetList::search(const std::string& val) {
    CoarseNode *pred, *curr;
    unsigned long key = std::hash<std::string>{}(val);
    pred = head;
    std::lock_guard<std::recursive_mutex> lk(lock);
    curr = pred->next;
    while (curr->key < key) {
        pred = curr;
        curr = curr->next;
    }
    return pred; 
}

bool CoarseSetList::add(const std::string& val) {
    std::lock_guard<std::recursive_mutex> lk(lock);
    CoarseNode* pred = this->search(val);
    CoarseNode* curr = pred->next;
    bool exists = (curr->key == std::hash<std::string>{}(val));
    if (!exists) {
        CoarseNode* node = new CoarseNode(val);
        node->next = curr;
        pred->next = node;
    }
    return !exists;
}

bool CoarseSetList::remove(const std::string& val) {
    std::lock_guard<std::recursive_mutex> lk(lock);
    CoarseNode* pred = this->search(val);
    CoarseNode* curr = pred->next;
    bool exists = (curr->key == std::hash<std::string>{}(val));
    if (exists) {
        pred->next = curr->next;
        delete curr;
    }
    return exists;
}

bool CoarseSetList::contains(const std::string& val) {
    std::lock_guard<std::recursive_mutex> lk(lock);
    CoarseNode* pred = this->search(val);
    CoarseNode* curr = pred->next;
    bool exists = (curr->key == std::hash<std::string>{}(val));
    return exists;
}

void CoarseSetList::print() const {
    CoarseNode* cur = this->head->next;
    while (cur->next != NULL) {
        std::cout << cur->item << " ";
        cur = cur->next;
    }
    std::cout << std::endl;
}
