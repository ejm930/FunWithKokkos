#include <Kokkos_Core.hpp>
#include <cstdio>


typedef Kokkos::TeamPolicy<> team_policy;
typedef team_policy::member_type team_member;
typedef Kokkos::DefaultExecutionSpace ExecSpace;
typedef Kokkos::TeamPolicy<ExecSpace>::member_type member_type;
    


int main(int narg, char* args[]) {
    Kokkos::initialize(narg, args);

    // Launch 12 teams of the maximum number of threads per team
    const int league_size = 32;

    // In practice it is often better to let Kokkos decide on the team_size
    //const team_policy policy_b(12, Kokkos::AUTO);

    //View<int> count("Count");
    Kokkos::TeamPolicy<ExecSpace> policy(league_size, 4);//Kokkos::AUTO());

    parallel_for(policy, KOKKOS_LAMBDA(member_type team_member) {
        //calc the global thread id
        int k = team_member.league_rank() * team_member.team_size() +
                team_member.team_rank();
        printf("LR TR GID: %i %i %i\n", team_member.league_rank(), team_member.team_rank(), k);
    });
    Kokkos::finalize();
}
