#include <iostream>
#include <thread>
#include <functional>
#include <mutex>
#include <condition_variable>

class Semaphore {
    int counter;
    std::mutex mtx;
    std::condition_variable cv;
    
public:
    Semaphore() { counter = 0; }
    Semaphore(int init_val) {
        counter = init_val;
    }
    
    void acquire() {
        std::unique_lock<std::mutex> lock(mtx);
        while (counter == 0) {
            cv.wait(lock);
        }   
        --counter;
    }
    
    void release() {
        std::unique_lock<std::mutex> lock(mtx);
        ++counter;
        cv.notify_one();
    }
};

Semaphore sem1, sem2;
    
void first() {
    
    // printFirst() outputs "first". Do not change or remove this line.
    std::cout << "first" << std::endl;
    sem1.release();
}

void second() {
    sem1.acquire();
    // printSecond() outputs "second". Do not change or remove this line.
    std::cout << "second" << std::endl;
    sem2.release();
}

void third() {
    sem2.acquire();
    // printThird() outputs "third". Do not change or remove this line.
    std::cout << "third" << std::endl;
}

int main()
{
    std::thread th1(first);
    std::thread th2(second);
    std::thread th3(third);
    
    th1.join();
    th2.join();
    th3.join();
    return 0;
}

