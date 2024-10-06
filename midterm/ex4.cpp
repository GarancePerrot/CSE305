#include <algorithm>
#include <climits>
#include <thread>
#include <numeric>
#include <iterator>
#include <vector>
#include <mutex>
#include <iostream>

//-----------------------------------------------------------------------------

template <unsigned int N>
class Train {
private:
    // The i-th item is the name of the passanger on the i-th seat. Empty string means that the seat is not reserved
    std::string passenger_name[N];
    // one lock per seat
    std::mutex locks[N];

public:
    Train() {
        for (size_t i = 0; i < N; ++i) {
            passenger_name[i] = "";
        }
    }

    std::string get_passenger(size_t seat) const {
        return passenger_name[seat];
    }

    // @brief Attempts to reserve a seat in an atomic manner
    // @param seat is the index of the seat (from 0 to N - 1)
    // @param name is the name of the person for the reservation
    // @returns true if the seat was succesfully reserved by this thread and false otherwise
    bool reserve(size_t seat, std::string name);

    // @brief Attempts to reserve two seats in an atomic manner
    // @param seat1, seat2 are the indices of the seat
    // @param name1 and name2 are the respective names
    // @returns true if and only if all the seats have been succesdully reserved. If at least one was not possible to reserve, 
    // does nothing and returns false
    bool reserve(size_t seat1, size_t seat2, std::string name1, std::string name2);


};

template <unsigned int N>
bool Train<N>::reserve(size_t seat, std::string name) {

    bool res = false;
    locks[seat].lock();
    if (passenger_name[seat] == ""){
        passenger_name[seat] = name;
        res = true;
    }
    locks[seat].unlock();
    return res;

}

template <unsigned int N>
bool Train<N>::reserve(size_t seat1, size_t seat2, std::string name1, std::string name2) {

    bool res = false;
    if (seat1 < seat2){
        std::lock_guard<std::mutex> lock_1(locks[seat1]);
        std::lock_guard<std::mutex> lock_2(locks[seat2]);
    } else if (seat2 < seat1){
        std::lock_guard<std::mutex> lock_2(locks[seat2]);
        std::lock_guard<std::mutex> lock_1(locks[seat1]);
    }

    if ((passenger_name[seat1] == "") && (passenger_name[seat2] == "")){
        passenger_name[seat1] = name1;
        passenger_name[seat2] = name2;
        res = true;
    }
    return res;

}



//-----------------------------------------------------------------------------

const size_t LENGTH = 2000;
const size_t RESERVATIONS = 1000;

void reserver(Train<LENGTH>& t, size_t seat, std::string name, size_t& count) {
    for (size_t i = 0; i < RESERVATIONS; ++i) {
        if (t.reserve(seat, name)) {
            ++count;
        }
    }
}

void reserver2(Train<LENGTH>& t, size_t seat1, size_t seat2, std::string name1, std::string name2, size_t& count) {
    for (size_t i = 0; i < RESERVATIONS; ++i) { 
        if (t.reserve(seat1, seat2, name1, name2)) {
            ++count;
        }
    }
}

int main(int argc, char* argv[]) {
    size_t countA = 0;
    size_t countB = 0;
    std::thread tA;
    std::thread tB;
    const size_t RUNS = 3000;

    std::cout << "Running many single reservations in parallel" << std::endl;
    bool failed = false;
    for (int i = 0; i < RUNS; ++i) {
        countA = 0;
        countB = 0;
        Train<LENGTH> t;

        tA = std::thread(&reserver, std::ref(t), 100, "A", std::ref(countA));
        tB = std::thread(&reserver, std::ref(t), 100, std::string("B"), std::ref(countB));
        tA.join();
        tB.join();

        if (countA + countB != 1) {
            std::cout << countA + countB << " reservations successful instead of exactly one --- NOT OK" << std::endl;
            failed = true;
            break;
        }

        if (!((countA > 0 && t.get_passenger(100) == "A") 
            || (countB > 0 && t.get_passenger(100) == "B"))) {
            failed = true;
            std::cout << "Reservation made for a wrong name --- NOT OK" << std::endl;
            break;
        }
    }
    if (!failed) {
        std::cout << "OK" << std::endl;
    }

    std::cout << "Running a large number of double reservations in parallel" << std::endl;
    failed = false;
    for (size_t i = 0; i < RUNS; ++i) {
        countA = 0;
        countB = 0;
        Train<LENGTH> t;

        tA = std::thread(&reserver2, std::ref(t), 42, 43, "A", "B", std::ref(countA));
        tB = std::thread(&reserver2, std::ref(t), 43, 42, "C", "D", std::ref(countB));
        tA.join();
        tB.join();

        if (countA + countB != 1) {
            std::cout << countA + countB << " reservations succeded instead of exactly one --- NOT OK" << std::endl;
            failed = true;
            break;
        }
 
        if (!((countA > 0 && t.get_passenger(42) == "A" && t.get_passenger(43) == "B") 
            || (countB > 0 && t.get_passenger(43) == "C" && t.get_passenger(42) == "D"))) {
            std::cout << "Reservation made for a wrong name --- NOT OK" << std::endl;
            failed = true;
            break;
        }
    }
    if (!failed) {
        std::cout << "OK" << std::endl;
    }


    std::cout << "Running a large number of reservations of both types in parallel" << std::endl;
    failed = false;
    for (size_t i = 0; i < RUNS; ++i) {
        countA = 0;
        countB = 0;
        Train<LENGTH> t;

        tA = std::thread(&reserver, std::ref(t), 1893, "A", std::ref(countA));
        tB = std::thread(&reserver2, std::ref(t), 1893, 1894, "B", "B", std::ref(countB));
        tA.join();
        tB.join();

        if (countA + countB != 1) {
            std::cout << countA + countB << " reservations succeded instead of exactly one --- NOT OK" << std::endl;
            failed = true;
            break;
        }
 
        if (!((countA > 0 && t.get_passenger(1893) == "A" && t.get_passenger(1894) == "") 
            || (countB > 0 && t.get_passenger(1893) == "B" && t.get_passenger(1893) == "B"))) {
            std::cout << "Reservation made for a wrong name --- NOT OK" << std::endl;
            failed = true;
            break;
        }
    }
    if (!failed) {
        std::cout << "OK" << std::endl;
    }
}
