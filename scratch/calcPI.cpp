#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstdio>

#include <chrono>

//class RandomPI {
//    public:
//        
//}
struct hello_world {
KOKKOS_INLINE_FUNCTION
void operator()(const int i) const { 
        printf("Hello from i = %i\n", i); 
        Kokkos::Random_XorShift64_Pool<> rand_pool64(666);
        typename Kokkos::Random_XorShift64_Pool<>::generator_type rand_gen = rand_pool64.get_state();
    
        auto uniform_sample = rand_gen.drand(0.0, 1.0);
        
        printf("Hello from i = %i and sample = %f\n", i, uniform_sample);  
        
        rand_pool64.free_state(rand_gen);

    }
};

//one way to do it, but maybe it might need its own cool random pool
template<class ExecSpace, class ViewType, class RandPoolType>
//void sample_random(ExecSpace ex, ViewType data, RandPoolType pool) {
void sample_random(ViewType data, RandPoolType pool) {
    //add check for random pool 
    static_assert(
        Kokkos::SpaceAccessibility<ExecSpace, typename ViewType::memory_space>::accessible,
        "Incompatible ViewType and ExecutionSpace"
    );

    Kokkos::parallel_for(
        "Sample_Random", 
        Kokkos::RangePolicy<ExecSpace>(ExecSpace(), 0, data.extent(0)), 
        KOKKOS_LAMBDA(const int i) {
            auto rand_gen = pool.get_state();
            auto uniform_sample = rand_gen.drand(0.0, 1.0);
            data(i) = uniform_sample;
            pool.free_state(rand_gen);
        }
    );
}

//one way to do it, but maybe it might need its own cool random pool
template<class ExecSpace, class ViewType, class RandPoolType>
//void sample_random(ExecSpace ex, ViewType data, RandPoolType pool) {
void sample_random2(ViewType data, RandPoolType pool) {
    //add check for random pool 
    static_assert(
        Kokkos::SpaceAccessibility<ExecSpace, typename ViewType::memory_space>::accessible,
        "Incompatible ViewType and ExecutionSpace"
    );

    Kokkos::parallel_for(
        "Sample_Random", 
        Kokkos::RangePolicy<ExecSpace>(ExecSpace(), 0, data.extent(0)), 
        KOKKOS_LAMBDA(const int i) {
            auto rand_gen = pool.get_state();
            auto uniform_sample = rand_gen.drand(0.0, 1.0);
            data(i) = uniform_sample;
            pool.free_state(rand_gen);
        }
    );
}

//compile time loops
template<int N>
void bobsfunction() {
    printf("\nN = %d \n", N);
    bobsfunction<N-1>();
}

template<>
void bobsfunction<0>() {
    printf("\nDone!\n");
}

template<class MyCoolType>
class myVector {
    size_t size;
    MyCoolType * arr;
    
    public:
        myVector(size_t length) {
            size = length;
            arr = new MyCoolType [length];
        }

        ~myVector() { delete [] arr; };

        MyCoolType operator[] (const size_t idx) {
            if(idx < size && idx > 0) {
                return arr[idx];
            } else {
                printf("Error, you suck!\n");
            }
        }
};

template<typename T1, typename T2, typename T3>
struct my_bag {
    T1 one; T2 two; T3 three;
    
    template<typename Type>
    Type operator[] (const size_t idx) const {
        if(idx == 0) {
            return one;
        }
        //etc
    }

    template<typename Type>
    Type operator[]<0> const {return one;}

    template<typename Type>
    Type operator[]<1> const {return two;}

    template<typename Type>
    Type operator[]<2> const {return three;}
    
    template<typename Type>
    Type operator[]<idx> const;

};

//template specializations
template<> 
class myVector<int> {
    //this is how kokkos range policy works!
};

//we can even do partial specialization!

//typedef Kokkos::View<double*, Kokkos::HostSpace> view_type;
typedef Kokkos::View<double*, Kokkos::HostSpace> view_type;

int main(int argc, char* argv[]) {
  
    Kokkos::ScopeGuard kokkos(argc, argv);

    //must intialized before any KOKKOS is called
    //Kokkos::initialize(argc, argv);

    printf("Default Execution Space: %s\n", typeid(Kokkos::DefaultExecutionSpace).name());

    auto SEED = std::chrono::system_clock::now().time_since_epoch().count();

    Kokkos::Random_XorShift64_Pool<Kokkos::HostSpace> rand_pool64(666);
    
    size_t N = 100;

    view_type randNums("Random Numbers", N);
    
    sample_random<Kokkos::DefaultHostExecutionSpace>(randNums, rand_pool64);
    //sample_random<Kokkos::Serial>(a, rand_pool64);

    for(auto i=0; i<N; i++) {
        printf("Entry %i has values: %f\n", i, randNums(i));
    }

    //Kokkos::finalize();
}
