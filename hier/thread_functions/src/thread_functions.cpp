#include <Kokkos_Core.hpp>
#include <cstdio>

void hello_world_verbose() { 
    size_t league_size = 5;
    const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy(league_size, Kokkos::AUTO());
    
    Kokkos::parallel_for("hello_world_verbose", policy, 
            KOKKOS_LAMBDA(Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type team_member) {
        
        int k = team_member.league_rank() * team_member.team_size() +
            team_member.team_rank();

        printf("On global thread %i\n", k);
    });
}

void hello_world_quiet() {
    typedef Kokkos::DefaultExecutionSpace ExecSpace;
    typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
    typedef team_policy::member_type member_type;
    
    size_t league_size = 5;
    const team_policy policy(league_size, Kokkos::AUTO());
    
    Kokkos::parallel_for("hello_world_quiet", policy, KOKKOS_LAMBDA(member_type team_member) {
        
        int k = team_member.league_rank() * team_member.team_size() +
            team_member.team_rank();

        printf("On global thread %i\n", k);
    });
}

void hello_world_custom(size_t league_size = 5, size_t team_size = 1) {
    
    typedef Kokkos::DefaultExecutionSpace ExecSpace;
    typedef Kokkos::TeamPolicy<ExecSpace> team_policy;
    typedef team_policy::member_type member_type;
    
    const team_policy policy(league_size, team_size);
    
    Kokkos::parallel_for("hello_world_custom", policy, KOKKOS_LAMBDA(member_type team_member) {
        
            Kokkos::parallel_for(Kokkos::)
        int k = team_member.league_rank() * team_member.team_size() +
            team_member.team_rank();

        printf("On global thread %i\n", k);
    });
}



int main(int narg, char* args[]) {
    
    Kokkos::ScopeGuard kokkos(narg, args);
    
    //hello_world_verbose();
    
    //hello_world_quiet();

    hello_world_custom();
    hello_world_custom(4, 2);

    return 0;
}
