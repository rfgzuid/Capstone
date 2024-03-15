import cProfile
import pendulum
def main():
    # Your main execution logic here
    pendulum.state_space_plot(200, 1)

cProfile.run('main()', 'profile_output4')