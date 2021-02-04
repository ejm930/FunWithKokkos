#include <Kokkos_Core.hpp>
#include <cstdio>


typedef Kokkos::TeamPolicy<> team_policy;
typedef team_policy::member_type team_member;

// Define a functor which can be launched using the TeamPolicy
struct hello_world {
  typedef int value_type;  // Specify value type for reduction target, sum

  // This is a reduction operator which now takes as first argument the
  // TeamPolicy member_type. Every member of the team contributes to the
  // total sum.
  // It is helpful to think of this operator as a parallel region for a team
  // (i.e. every team member is active and will execute the code).
  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member& thread, int& sum) const {
    sum += 1;
    // The TeamPolicy<>::member_type provides functions to query the multi
    // dimensional index of a thread as well as the number of thread-teams and
    // the size of each team.
    printf("Hello World: %i %i // %i %i\n", thread.league_rank(),
           thread.team_rank(), thread.league_size(), thread.team_size());
  }
};

int main(int narg, char* args[]) {
Kokkos::initialize(narg, args);

    // Launch 12 teams of the maximum number of threads per team
    const int team_size_max = team_policy(1, 1).team_size_max(
      hello_world(), Kokkos::ParallelReduceTag());
    const team_policy policy_a(12, team_size_max);

    int sum = 0;
    Kokkos::parallel_reduce(policy_a, hello_world(), sum);

    // The result will be 12*team_size_max
    printf("Result A: %i == %i\n", sum, team_size_max * 12);

    // In practice it is often better to let Kokkos decide on the team_size
    const team_policy policy_b(12, Kokkos::AUTO);

    Kokkos::parallel_reduce(policy_b, hello_world(), sum);
    // The result will be 12*policy_b.team_size_recommended( hello_world(),
    // Kokkos::ParallelReduceTag())
    const int team_size_recommended = policy_b.team_size_recommended(
      hello_world(), Kokkos::ParallelReduceTag());
    printf("Result B: %i %i\n", sum, team_size_recommended * 12);

    int sum2 = 0;
    Kokkos::parallel_reduce(policy_a, KOKKOS_LAMBDA(const team_member &thread, int &update) {
          update += 1;
          printf("Hello World: %i %i // %i %i\n", thread.league_rank(),
           thread.team_rank(), thread.league_size(), thread.team_size());
    }, sum2);
    printf("Sum: %i\n", sum2);

    //we could make the execution space arbitrary :)
    Kokkos::RangePolicy<Kokkos::Serial> policy_1(Kokkos::Serial(), 2, 6);
    Kokkos::parallel_for("Loop", policy_1, KOKKOS_LAMBDA(const int i) {
    printf("On loop %i\n", i);
    });

    //Let's mess around with some more team stuff
    using Kokkos::TeamPolicy;
    using Kokkos::parallel_for;
    using Kokkos::View;
    typedef Kokkos::DefaultExecutionSpace ExecSpace;
    typedef TeamPolicy<ExecSpace>::member_type member_type;
    
    //View<int> count("Count");
    size_t league_size = 5;
    TeamPolicy<ExecSpace> policy(league_size, Kokkos::AUTO());

    parallel_for(policy, KOKKOS_LAMBDA(member_type team_member) {
        //calc the global thread id
        int k = team_member.league_rank() * team_member.team_size() +
                team_member.team_rank();
        //calculate the sum of the global thread ids of this team
        //int team_sum = team_member.team_reduce(k);
    });
    Kokkos::finalize();
}
