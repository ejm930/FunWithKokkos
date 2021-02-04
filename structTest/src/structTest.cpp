#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <cstdio>
#include <chrono>

struct my_pipeline {
    public:
    using Space=Kokkos::DefaultExecutionSpace;
    using eric_t=Kokkos::View<float *, Space>;

    eric_t vect;
    size_t N;
    float factor;

    my_pipeline(size_t _N, float _factor) : N(_N), factor(_factor) {}
    
    
    void dispatch() {
        vect = eric_t("vect", N);
        
        printf("Vector size: %zu\n", N);
        
        //fill the vect in parallel on the device
        auto const &vect_ = vect;
        auto const &factor_ = factor;
        Kokkos::parallel_for("my_pipeline::vect_fill",
        Kokkos::RangePolicy<Space>(0, N),
        KOKKOS_LAMBDA(int i) {
            vect_(i) = multiply((float)i);
        });
    }

    KOKKOS_INLINE_FUNCTION
        float multiply(float i) {
            return factor_ * i;
        }
};
/* only works with cuda 8.0 or higher and cxx17+
class my_fancy {
    public:
    using Space=Kokkos::DefaultExecutionSpace;
    using eric_t=Kokkos::View<float *, Space>;

    eric_t vect;
    size_t N;

    my_fancy(size_t _N) : N(_N) {}
    
    
    void dispatch() {
        vect = eric_t("vect", N);
        
        printf("Vector size: %zu\n", N);
        
        //fill the vect in parallel on the device
        Kokkos::parallel_for("my_pipeline::vect_fill",
        Kokkos::RangePolicy<Space>(0, N),
        KOKKOS_CLASS_LAMBDA(int i) {
            vect(i) = i;
        });
    }
};*/



int main(int argc, char* argv[]) {
  
    Kokkos::ScopeGuard kokkos(argc, argv);

    printf("\n\nDefault Execution Space: %s\n\n", typeid(Kokkos::DefaultExecutionSpace).name());
    
    struct my_pipeline erock(10, 2.0);
    erock.dispatch();
    Kokkos::fence();

    return 0;
}

/*template<class ExecSpace, class MemSpace>
size_t sample_PI(size_t num_samples) {
    
    //could add fancy checks

    auto SEED = std::chrono::system_clock::now().time_since_epoch().count();

    Kokkos::Random_XorShift64_Pool<MemSpace> pool(SEED);
    
    size_t hits = 0;

    Kokkos::parallel_reduce(
        "Sample_PI", 
        Kokkos::RangePolicy<ExecSpace>(ExecSpace(), 0, num_samples), 
        KOKKOS_LAMBDA(const int i, size_t &update) {
            auto rand_gen = pool.get_state();
            //uniformly distributed random samples
            auto x = rand_gen.drand(0.0, 1.0);
            auto y = rand_gen.drand(0.0, 1.0);
            
            if(x*x + y*y < 1.0) {
                update++;
            }
            pool.free_state(rand_gen);
    },hits);

    return hits;
}

double calculatePI(size_t hits, size_t num_samples) {
    return 4.0*static_cast<double>(hits)/static_cast<double>(num_samples);
}

double computeRErr(double true_val, double approx_val) {
    return abs(true_val - approx_val)/true_val;
}

template<class ExecSpace, class MemSpace>
void run_sim(size_t num_samples) {
    
    printf("Sampling %zu points using the exection space %s\n", num_samples, typeid(ExecSpace).name()); 
    
    auto t1 = std::chrono::high_resolution_clock::now();
    size_t hits = sample_PI<ExecSpace, MemSpace>(num_samples);
    Kokkos::fence();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);

    double true_val = calculatePI(hits, num_samples);
    double rel_err  = computeRErr(true_val, M_PI);
    
    printf("True value of PI: %f, Approximate Value: %f, Relative Error: %f\n", M_PI, true_val, rel_err);
    printf("Total Time In Kernel: %f seconds\n\n", time_span.count());
}

int main(int argc, char* argv[]) {
  
    //we do this to avoid having to call Kokkos::initialze() and 
    //Kokkos::finalize(). No need for nested bracing.
    Kokkos::ScopeGuard kokkos(argc, argv);

    printf("\n\nDefault Execution Space: %s\n\n", typeid(Kokkos::DefaultExecutionSpace).name());
   
    size_t num_samples = 100000000; 
    run_sim<Kokkos::Serial, Kokkos::HostSpace>(num_samples);

    run_sim<Kokkos::OpenMP, Kokkos::HostSpace>(num_samples);

    run_sim<Kokkos::Cuda, Kokkos::CudaSpace>(num_samples);

    return 0;
}*/
