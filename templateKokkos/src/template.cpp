#include <Kokkos_Core.hpp>
#include <cstdio>

KOKKOS_INLINE_FUNCTION
void collide_child(int const& i) { 
    printf("Hello from collide %i\n", i);
}


template<bool MonteCarlo>
struct BinaryCollision {
    KOKKOS_INLINE_FUNCTION
    void collide(int const& i) { 
        printf("Hello from collide %i\n", i);
    }
};
template<bool MonteCarlo>
struct VoxelParallel : BinaryCollision<MonteCarlo> {
    using BinaryCollision<MonteCarlo>::collide;
    using Space = Kokkos::DefaultExecutionSpace;
    void apply_model() {
        printf("Applying Voxel Model\n");
        Kokkos::parallel_for("apply_model", Kokkos::RangePolicy<Space>(0, 2), KOKKOS_LAMBDA(const int i) {
            collide_child(i); 
        });
    } 
};
template<typename ParallelPolicy = VoxelParallel <true> >
struct CollisionParallelismModel : ParallelPolicy {
    using ParallelPolicy::apply_model;
};


template<typename ParallelPolicy = VoxelParallel <true> >
struct binary_collision_pipeline {
    CollisionParallelismModel< ParallelPolicy > pm;

    void dispatch() {
        pm.apply_model();
    }
};

int main(int argc, char* argv[]) {
  
    Kokkos::ScopeGuard kokkos(argc, argv);
    binary_collision_pipeline<VoxelParallel<true>> bcp;
    bcp.dispatch();
    return 0;
}
