#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <cstdio>
#include <chrono>

template<typename DataType>
double get_alloc_size(size_t num_elems) {
    double chunk_size = static_cast<double>(num_elems)*static_cast<double>(sizeof(DataType));
    
    //return size in gigabytes
    return chunk_size/1000000000.0;
}

template<class ViewTypeDefault, class ViewTypeMirror>
void copy_sim(ViewTypeDefault a, ViewTypeMirror b) {
    
    printf("Copying from %s to %s\n", typeid(ViewTypeDefault).name(), typeid(ViewTypeMirror).name());
    auto t1 = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(b, a);
    Kokkos::fence();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    printf("Total Time In Kernel: %f seconds\n\n", time_span.count());
}
int main(int argc, char* argv[]) {
  
    //we do this to avoid having to call Kokkos::initialze() and 
    //Kokkos::finalize(). No need for nested bracing.
    Kokkos::ScopeGuard kokkos(argc, argv);

    printf("\n\nDefault Execution Space: %s\n\n", typeid(Kokkos::DefaultExecutionSpace).name());
   
    size_t num_elems = 200000000; 
    
    Kokkos::View<double*> a("A", num_elems);
    Kokkos::View<double*>::HostMirror b = Kokkos::create_mirror_view(a);
    Kokkos::View<double*> c("C", num_elems);

    printf("Size of allocation: %f gb\n", get_alloc_size<double>(num_elems));
    
    copy_sim(a, b);
    copy_sim(b, a);
    copy_sim(a, c);

    return 0;
}
