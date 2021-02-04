#include <Kokkos_Core.hpp>
#include <cstdio>


typedef Kokkos::TeamPolicy<> team_policy;
typedef team_policy::member_type team_member;

int main(int narg, char* args[]) {
Kokkos::ScopeGuard kokkos(narg, args);

    using Space=Kokkos::DefaultExecutionSpace;
    using member_type=Kokkos::TeamPolicy<Space>::member_type;
 
    Kokkos::parallel_for("thread_range",
    Kokkos::TeamPolicy<Space>(2, 3/*Kokkos::AUTO()*/),
    KOKKOS_LAMBDA (member_type thread) {
        printf("Hello World: %i %i // %i %i\n", thread.league_rank(),
           thread.team_rank(), thread.league_size(), thread.team_size());

        int remain = 4;

        Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, remain),
        [&](const int& k) {

            printf("Thread Range: %i %i %i\n", thread.league_rank(), thread.team_rank(), k);        
        });
    });

    return 0;
}
