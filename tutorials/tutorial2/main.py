from environments.grid_world import SimpleGridWorld
from tutorials.tutorial2.policy_evaluator import PolicyEvaluator
from tutorials.tutorial2.policy_iteration_agent import PolicyIterationAgent
from tutorials.tutorial2.value_iteration_agent import ValueIterationAgent

if __name__ == "__main__":
    # Instantiate the environment
    env = SimpleGridWorld(max_steps=50)
    print("SimpleGridWorld Environment:")
    env.render()

    # Exercise 5: Iterative Policy Evaluation
    evaluator = PolicyEvaluator(env)
    v_policy_eval = evaluator.evaluate()
    print("Policy Evaluation - Value Function:")
    print(v_policy_eval)
    print()

    # Exercise 6: Policy Iteration
    pi_agent = PolicyIterationAgent(env)
    optimal_policy_pi, v_pi = pi_agent.iterate_policy()
    print("Policy Iteration - Optimal Policy (0=Up, 1=Down, 2=Left, 3=Right):")
    print(optimal_policy_pi)
    print("Associated Value Function:")
    print(v_pi)
    print()

    # Exercise 7: Value Iteration
    vi_agent = ValueIterationAgent(env)
    optimal_policy_vi, v_vi = vi_agent.iterate_value()
    print("Value Iteration - Optimal Policy:")
    print(optimal_policy_vi)
    print("Associated Value Function:")
    print(v_vi)