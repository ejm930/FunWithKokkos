#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <cstdio>
#include <chrono>

/*Usage*********************************************************************
auto SEED = std::chrono::system_clock::now().time_since_epoch().count();
Kokkos::Random_XorShift64_Pool<Kokkos::HostSpace> rand_pool64(666);
    
size_t N = 100;
view_type_host randNums("Random Numbers", N);
sample_random<Kokkos::DefaultHostExecutionSpace>(randNums, rand_pool64);
***************************************************************************/ 
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
template<class ExecSpace, class MemSpace>
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

//typedef Kokkos::View<double*, Kokkos::HostSpace> view_type_host;
//typedef Kokkos::View<double*, Kokkos::Cuda> view_type_cuda;

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
}
