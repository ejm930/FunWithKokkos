#include<iostream>
#include<string>
using namespace std;

struct DefaultSort {
    
    void sort() {
        std::cout << "Normal sort\n";
    }

    void shuffle_sort() {
        std::cout << "Using shuffle Sort\n";
    }

};

enum class ItemList {one, two, three};

template <typename Policy = DefaultSort>
struct ParticleSorter : private Policy {
      using Policy::sort;
      using Policy::shuffle_sort; 
};

class Other {

};
class BinaryOps {
    public:
        string myName;

        BinaryOps(string _myName) : myName(_myName) { }
        
        template<class T2, class T1>
        T1 add(T1 n1, T1 n2) {
            ParticleSorter<T2> stuff;
            return n1 + n2;
        }

        template<class partype>
        void message() {
            std::cout << "Hello from" << static_cast<int>(2.0) << "\n";
        }
};

int main() {
    BinaryOps myOp("Add");
    int a = 2; int b = 3;
    int c = myOp.add<DefaultSort>(a,b);
    std::cout << c << std::endl;
    myOp.message<ItemList::one>();
    return 0;
}
