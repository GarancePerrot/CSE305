#include <algorithm>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "CoarseSetList.cpp"
#include "SetList.cpp"

//-----------------------------------------------------------------------------


class BoundedSetList: public SetList {
    std::condition_variable not_full;
    size_t capacity;
    size_t count;
    std::mutex count_lock;
public:
    BoundedSetList(size_t capacity) {
        this->capacity = capacity;
        this->count = 0;
    }

    size_t get_capacity() const {return this->capacity;}
    size_t get_count() const {return this->count;}

    bool add(const std::string& val);
    bool remove(const std::string& val);
};

bool BoundedSetList::add(const std::string& val) {

    while (true) {
        Node* pred = this->search(val);
        Node* curr = pred->next;
        bool exists = (curr->key == std::hash<std::string>{}(val));
        if (exists) {  // element already exists
            pred->lock.unlock();
            curr->lock.unlock();
            return false;
        }

        std::unique_lock<std::mutex> lock(count_lock);
        if (count >= capacity) { //shoud wait, has reached capacity
        // should unlock everything and wait for the empty space to appear
            pred->lock.unlock();
            curr->lock.unlock();
            not_full.wait(lock);
            continue; 
        }

        //actual insertion
        Node* node = new Node(val);
        node->next = curr;
        pred->next = node;
        count++;
        lock.unlock();
        pred->lock.unlock();
        curr->lock.unlock();
        return true;
    }

}

bool BoundedSetList::remove(const std::string& val) {
    Node* pred = this->search(val);
    Node* curr = pred->next;
    bool exists = (curr->key == std::hash<std::string>{}(val));
    curr->lock.unlock();
    if (exists) {
        pred->next = curr->next;
        delete curr;
        std::unique_lock<std::mutex> lock(count_lock);
        count--;
        lock.unlock();
        not_full.notify_one();
    }
    pred->lock.unlock();
    return exists;
}

//-----------------------------------------------------------------------------
