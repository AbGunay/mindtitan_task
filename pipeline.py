import luigi
from model import Model


class Prediction(luigi.Task):
    #file_path = luigi.Parameter()

    def requires(self):
        return []

    def output(self):
        return luigi.LocalTarget("predictions.txt")

    def run(self):
        predictions = Model().get_prediction()
        with self.output().open('wb') as f:
            f.write("{}\n".format(predictions))


if __name__ == '__main__':
    luigi.run()
