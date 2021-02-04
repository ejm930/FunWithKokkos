#include <Kokkos_Core.hpp>
#include <cstdio>

int main(int narg, char* args[]) {
Kokkos::initialize(narg, args);
    
    typedef Kokkos::DefaultExecutionSpace ExecSpace;
    using team_policy = Kokkos::TeamPolicy<ExecSpace>;
    typedef team_policy::member_type tm;

    int league_size = 4;
    const team_policy policy_a(league_size, Kokkos::AUTO);

    int sum = 0;
    Kokkos::parallel_reduce(policy_a, KOKKOS_LAMBDA(const tm &thread, int &update) {
          update += 1;
          printf("Hello World: %i %i // %i %i\n", thread.league_rank(),
           thread.team_rank(), thread.league_size(), thread.team_size());
    }, sum);
    printf("Sum: %i\n", sum);

    //we could make the execution space arbitrary :)
    Kokkos::RangePolicy<Kokkos::Serial> policy_1(Kokkos::Serial(), 2, 6);
    Kokkos::parallel_for("Loop", policy_1, KOKKOS_LAMBDA(const int i) {
    printf("On loop %i\n", i);
    });
    
    league_size = 5;
    const team_policy policy_b(league_size, Kokkos::AUTO());
    parallel_for(policy_b, KOKKOS_LAMBDA(tm team_member) {
        //calc the global thread id
        int k = team_member.league_rank() * team_member.team_size() +
                team_member.team_rank();
        printf("Global %d\n", k);
    });
    Kokkos::finalize();
}
