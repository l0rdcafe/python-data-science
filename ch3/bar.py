from matplotlib import pyplot as plt

movies = ['Annie Hall', 'Ben-Hur', 'Casablanca', 'Gandhi', 'West Side Story']
n_oscars = [5, 11, 3, 8, 10]

plt.bar(range(len(movies)), n_oscars)
plt.title('My Favorite Movies')
plt.ylabel('# of Academy Awards')

plt.xticks(range(len(movies)), movies)
plt.show()
