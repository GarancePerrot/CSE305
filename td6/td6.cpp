#pragma once
#include <cfloat>
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <vector>
#include <condition_variable>
#include <atomic>

//-----------------------------------------------------------------------------

class Account {
        unsigned int amount;
        unsigned int account_id;
        unsigned int max_amount;
        std::mutex lock;
        // You may want to add condition variable(s) here
        std::condition_variable cvW;
        std::condition_variable cvA;
        

        static std::atomic<unsigned int> max_account_id;
    public:
        static const unsigned int DEFAULT_MAX_AMOUNT = 5000000;

        Account() {
            this->amount = 0;
            this->max_amount = DEFAULT_MAX_AMOUNT;
            this->account_id = max_account_id.fetch_add(1);
        }

        Account(unsigned int amount) {
            this->amount = amount;
            this->max_amount = DEFAULT_MAX_AMOUNT;
            this->account_id = max_account_id.fetch_add(1);
        }

        // copy-contructor and assignment are deleted to make the id's really unique
        Account(const Account& other) = delete;

        Account& operator = (const Account& other) = delete;
        
        unsigned int get_amount() const {
            return this->amount;
        }

        unsigned int get_id() const {
            return this->account_id;
        }

        unsigned int get_max_amount() const {
            return this->max_amount;
        }

        // withdrwas deduction (blocked if the current amount is less than deduction)
        void withdraw(unsigned int deduction) {
            std::unique_lock<std::mutex> lk(lock);
            while(get_amount() < deduction){  // operation not possible (the amount will get below zero)
                cvW.wait(lk);
            }
            amount -= deduction;
            cvA.notify_all();
        }

        // adds the prescribed amount of money to the account
        void add(unsigned int to_add) {
            std::unique_lock<std::mutex> lk(lock);
            while(get_amount() + to_add > get_max_amount()){ // operation not possible (the amount will get beyond the bound)
                cvA.wait(lk);
            }
            amount += to_add;
            cvW.notify_all();
        }

        bool change_max(unsigned int new_max) {
            std::unique_lock<std::mutex> lk(lock);
            if (get_amount() > new_max){
                return false;
            }
            max_amount = new_max;
            cvA.notify_all(); //in case some threads are waiting on adds
            return true;
        }

        static void transfer(Account& from, Account& to, unsigned int amount) {
            return ;
            if (from.get_id() < to.get_id()){
                std::lock_guard<std::mutex> lka(from.lock);
                std::lock_guard<std::mutex> lkb(to.lock);

                if (from.get_amount() < amount){ // check withdraw condition for 'from'
                    return;
                }

                if (to.get_amount() + amount > to.get_max_amount()){  // check add condition for 'to'
                    return;
                }

                from.amount -= amount;
                to.amount += amount; 
                from.cvW.notify_all();
                from.cvA.notify_all();
                to.cvW.notify_all();
                to.cvA.notify_all();

            } else {

                std::lock_guard<std::mutex> lka(to.lock);
                std::lock_guard<std::mutex> lkb(from.lock);

                if (from.get_amount() < amount){ // check withdraw condition for 'from'
                    return;
                }

                if (to.get_amount() + amount > to.get_max_amount()){  // check add condition for 'to'
                    return;
                }

                from.amount -= amount;
                to.amount += amount; 
                from.cvW.notify_all();
                from.cvA.notify_all();
                to.cvW.notify_all();
                to.cvA.notify_all();
            }

        }
};

std::atomic<unsigned int> Account::max_account_id(0);

//-----------------------------------------------------------------------------

class CoarseNode {
public:
    long key;
    CoarseNode* left;
    CoarseNode* right;
    CoarseNode* parent;
    CoarseNode() {}
    CoarseNode(long k) {
        this->key = k;
        this->left = NULL;
        this->right = NULL;
        this->parent = NULL;
    }
};

class CoarseBST {
protected:
    std::mutex lock;
    CoarseNode* root;
    static const unsigned long LOWEST_KEY = LONG_MIN;
public:
    CoarseBST() {
        this->root = new CoarseNode(CoarseBST::LOWEST_KEY);
    }
    ~CoarseBST();
    bool add(long k);
    bool contains(long k);
};

void DeleteTree(CoarseNode* root) {
    if (root->left != NULL) {
        DeleteTree(root->left);
    }
    if (root->right != NULL) {
        DeleteTree(root->right);
    }
    delete root;
}

CoarseBST::~CoarseBST() {
    std::lock_guard<std::mutex> lk(lock);
    DeleteTree(this->root);
}


bool CoarseBST::add(long k) {

    std::lock_guard<std::mutex> lk(lock); 
    CoarseNode* curr = root;
    CoarseNode* pred = nullptr;

    while (curr != nullptr) {  // finding the position to insert
        pred = curr;
        if ( k == curr->key){
            return false;
        }
        if (curr->key < k) {
            curr = curr->right;
        } 
        else {
            curr = curr->left;
        } 
    }

    CoarseNode* node = new CoarseNode(k); //creating a node to add
    node->parent = pred;
    if (pred == nullptr) { 
        root = node;  
    } 
    else if (pred->key < k) {
        pred->right = node;
    } 
    else {
        pred->left = node;
    }

    return true;
}

bool CoarseBST::contains(long k) {
    std::lock_guard<std::mutex> lk(lock);  
    CoarseNode* curr = root;

    while (curr != nullptr) { // traversig the tree
        if (k == curr->key){ // found kee
            return true;
        }
        if (curr->key < k) {
            curr = curr->right;
        } 
        else{
            curr = curr->left;
        } 
    }
    return false;
}

