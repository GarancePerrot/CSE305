#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <queue>

//----------------------------------------------------------------------------

class QueueException {
    std::string str;
public:
    QueueException(const std::string& s) : str(s) {}
    ~QueueException() {}
};

template <class E> 
class SafeUnboundedQueueLock {
        std::queue<E> elements;
        std::mutex lock;
    public: 
        SafeUnboundedQueueLock<E>(){}
        void push(const E& element);
        E pop_nonblocking();
        E pop_blocking(); // Exercise 2
        bool is_empty() const {return this->elements.empty();}
};

template <class E>
void SafeUnboundedQueueLock<E>::push(const E& element) {
    std::lock_guard<std::mutex> guard(lock);
    elements.push(element);
}


template <class E> 
E SafeUnboundedQueueLock<E>::pop_nonblocking() {
    std::lock_guard<std::mutex> guard(lock);
    if (is_empty()){
        throw QueueException("Empty queue");
    }
    E element = elements.front();
    elements.pop();
    return element;
}

template <class E> 
E SafeUnboundedQueueLock<E>::pop_blocking() {
    // Exercise 2
    std::unique_lock<std::mutex> ulock(lock);
    //does not throw an exception but uses busy waiting: if queue is empty,
    // it will release the lock and try again until it succeeds
    while (is_empty()) {
      ulock.unlock();
      ulock.lock();
    }
    E element = elements.front();
    elements.pop();
    return element;
}

//----------------------------------------------------------------------------


void DivideOnceEven(std::condition_variable& iseven, std::mutex& m, int& n) {

    std::unique_lock<std::mutex> lk(m);
    while (n % 2 != 0) {
        iseven.wait(lk); // uses the lock to wait on the cv until n is even
    }
    n /= 2; 
}

//-----------------------------------------------------------------------------

template <class E> 
class SafeUnboundedQueueCV {
        std::queue<E> elements;
        std::mutex lock;
        std::condition_variable not_empty;
    public: 
        SafeUnboundedQueueCV<E>(){}
        void push(const E& element);
        E pop ();
        bool is_empty() const {return this->elements.empty();}
};

template <class E>
void SafeUnboundedQueueCV<E>::push(const E& element) {
    std::lock_guard<std::mutex> guard(lock);
    elements.push(element);
    not_empty.notify_one();  // wakes up any thread waiting in pop()
}

template <class E> 
E SafeUnboundedQueueCV<E>::pop() {

    std::unique_lock<std::mutex> lk(lock);
    while (is_empty()) {
        not_empty.wait(lk);
    }
    E element = elements.front();
    elements.pop();
    return element;
}

//-----------------------------------------------------------------------------





