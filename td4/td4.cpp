#pragma once
#include <cfloat>
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <vector>
#include <atomic>
#include <iostream>
#include <mutex>
#include <algorithm>
#include <iostream>
#include <chrono>
//-----------------------------------------------------------------------------

template <typename T>
void FindThread(T* arr, size_t block_size, T target, unsigned int count, std::atomic<unsigned int>& occurences) {

    for (int i=0 ; i<block_size; ++i){
        if (arr[i] == target){
            // increment the atomic int occurences

            // while (true) {
            //     n = occurences.load() ;
            //     if (occurences.compare_exchange_weak(n, n+1)) {
            //         //and compare it with count 
            //         if (occurences >= count){ // checking >= if multiple threads act simultaneously
            //             return;
            //         }
            //         break;
            //     }  
            // }
            occurences.fetch_add(1);
        
        }
        if (occurences >= count){
            return;
        }
    }  
}

/**
 * @brief Checks if there are at least `count` occurences of targert in the array
 * @param arr - pointer to the first element of the array
 * @param N - the length of the array
 * @param target - the target to search for
 * @param count - the number of occurences to stop after
 * @param num_threads - the number of threads to use
 * @return if there are at least `count` occurences
*/
template <typename T>
bool FindParallel(T* arr, size_t N, T target, size_t count, size_t num_threads) {

    if (N == 0) {
        return false;
    }

    std::atomic<unsigned int> occurences(0);
    size_t block_size = N / num_threads;
    std::vector<std::thread> workers(num_threads - 1);
    T* start = arr; 

    for (size_t i = 0; i < num_threads - 1; ++i) {
        workers[i] = std::thread(&FindThread<T>, start, block_size, target, count, std::ref(occurences));
        start += block_size; 
    }
    size_t remaining_size =  N - (start - arr) ; 
    FindThread(start, remaining_size , target, count, std::ref(occurences));

    for (size_t i = 0; i < num_threads - 1; ++i) {
        workers[i].join();
    }

    if (occurences >= count){
        return true; 
    }

    return false;
}



//-----------------------------------------------------------------------------

class Account {
        unsigned int amount;
        unsigned int account_id;
        std::mutex lock;

        static std::atomic<unsigned int> max_account_id;
    public:
        Account() : Account(0) {
        }

        Account(unsigned int amount) {

            //initializing class members
            this->lock.lock();
            this->amount = amount;
            //max_account_id.fetch_add(1);
            //here chatgpt helped me to avoid the problem of "Parallel creation of accounts yields equal ids" generated with the max_account_id incrementation
            this->account_id = max_account_id++; 
            this->lock.unlock();

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

        // withdrwas deduction if the current amount is at least deduction
        // returns whether the withdrawal took place
        bool withdraw(unsigned int deduction) {

            bool res = false;
            this->lock.lock();

            //checking if the funds are sufficient
            if (this->amount>= deduction) {
                this->amount -= deduction;
                res = true;
            }
            this->lock.unlock();
            return res;
        }

        // adds the prescribed amount of money to the account
        void add(unsigned int to_add) {
            this->lock.lock();
            this->amount += to_add;
            this->lock.unlock();
        }

        // transfers amount from from to to if there are enough money on from
        // returns whether the transfer happened
        static bool transfer(unsigned int amount, Account& from, Account& to) {
            
            // I TRIED THIS APPROACH BUT I HAD A PROBLEM OF DEADLOCK 

            // bool res = false;
            // bool flag = false;
            // if (from.account_id < to.account_id){
            //     while (!flag){
            //         if (from.lock.try_lock()){
            //             if (to.lock.try_lock()){
            //                 // we check if the withdrawal from 'from' happened before adding to 'to'
            //                 if (from.withdraw(amount)){
            //                     to.add(amount);
            //                     res = true;
            //                 }
            //                 flag = true;
            //                 to.lock.unlock();
            //             }
            //             from.lock.unlock();  
            //         }
            //     }
            // }

            // else if (from.account_id > to.account_id){
            //     while(!flag){
            //         if (to.lock.try_lock()){
            //             if (from.lock.try_lock()){
            //                 // we check if the withdrawal from 'from' happened before adding to 'to'
            //                 if (from.withdraw(amount)){
            //                     to.add(amount);
            //                     res = true;
            //                 }
            //                 flag = true;
            //                 from.lock.unlock();
            //             }
            //             to.lock.unlock();
            //         }
            //     }
            // }
            // return res;

            bool res = false;
            if (from.get_id() < to.get_id()) {
                std::lock_guard<std::mutex> lock_from(from.lock);
                std::lock_guard<std::mutex> lock_to(to.lock);
   
            } else if(from.get_id() > to.get_id()) {
                std::lock_guard<std::mutex> lock_to(to.lock);
                std::lock_guard<std::mutex> lock_fromo(from.lock);

            }

            // Critical section 
            if (from.withdraw(amount)) {
                to.add(amount);
                res = true;
            }
            return res;
        }
        
        //auxiliary function to sort Accounts by ids
        static bool aux_sort(const Account* a1, const Account* a2){
            if (a1->get_id() < a2->get_id()){
                return true;
            }
            return false;
        }

        static bool massiv_withdraw(std::vector<Account*>& accounts, unsigned int amount) {

            size_t n = accounts.size();

            // we start by sorting the accounts by ids
            std::sort(accounts.begin(), accounts.end(), aux_sort);

            // then acquire the locks for each account (using unique_lock)
            std::vector<std::unique_lock<std::mutex>> locks;
            for (int i = 0; i< n; i++){
                locks.push_back(std::unique_lock<std::mutex>(accounts[i]->lock));
            }

            //checking if the funds are sufficient for each account
            bool all_good = true;
            for (int i = 0; i< n; i++){

                if (accounts[i]->get_amount() < amount){
                    all_good = false;
                    break; //if the condition is false for one account, no need to check for the rest
                }     
            }

            //finally we perform the transaction if all good
            if (all_good){
                for (int i = 0; i< n; i++){
                    accounts[i]->amount -= amount;
                }
            }
            return all_good;
        }

};

std::atomic<unsigned int> Account::max_account_id(0);

//-----------------------------------------------------------------------------

class MyLock {

    private:
        std::atomic<bool> is_locked;
        std::thread::id my_id;

    public:

        MyLock() {
            is_locked = false;
            my_id = std::this_thread::get_id();
        }

        void lock() {
            //if another thread has already locked the mutex, we block execution until the lock is acquired
            bool expected = false;
            while (!is_locked.compare_exchange_weak(expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
                expected = false;
            }
            my_id = std::this_thread::get_id();
        }
        
        // specification as https://en.cppreference.com/w/cpp/thread/mutex/lock
        // but no need to worry about exceptions

        bool try_lock() { 
            // On successful lock acquisition returns true, otherwise returns false
            bool expected = false;
            // if (std::this_thread::get_id() == my_id) {
            //     return is_locked.compare_exchange_weak(expected, true, std::memory_order_acquire, std::memory_order_relaxed);
            // }
            return is_locked.compare_exchange_weak(expected, true, std::memory_order_acquire, std::memory_order_relaxed);
        }
        // specification as https://en.cppreference.com/w/cpp/thread/mutex/try_lock


        void unlock() {
            //The mutex must be locked by the current thread of execution, otherwise, the behavior is undefined
            if  ((std::this_thread::get_id() == my_id) && (is_locked)){
                is_locked = false;
                return;
            }
        }
        // specification as https://en.cppreference.com/w/cpp/thread/mutex/unlock
        // but if called by a nonowner thread, nothing should happen
};

//-----------------------------------------------------------------------------


