# create a new class called TemporalUnwrap

from os import fdopen


class NwpuGradstudent:

    def __init__(self, height, weight, age, gender) -> None:
        self.height = height
        self.weight = weight
        self.age = age
        self.gender = gender

    @property
    def graduate_year(self):
        return self.age + 4


    def get_graduate_year(self):
        return self.age + 4

    @property
    def bmi(self):
        return self.weight / (self.height ** 2)

    def _shower(self):
        print("Showering!")

    def _eat(self):
        if self.weight > 70:
            print("Eating More!")
        else:
            print("Eating Less!")

    def _run(self):
        if self.gender == "Male":
            print("Runn 10 laps!")
        elif self.gender == "Female":
            print("Runn 8 laps!")

    def exercise(self):
        self._eat()
        self._run()
        self._shower()

        return self


if __name__ == "__main__":

    par_file = fopen("par_file.txt", "r")
    for par in par_file:
        test_periodogram = NwpuGradstudent(**par)
        test_periodogram.simulate




