import cProfile
import main
cProfile.run('main.evaluate(main.env)', 'profile_output')
