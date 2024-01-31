from metaflow import (
    FlowSpec,
    Parameter,
    project,
    step,
    card
)

import os

def install_dependencies():
    os.system('pip install pandas~="2.0.1"')
    # os.system(
    #     "pip install -i https://northamerica-northeast1-python.pkg.dev/prolaio-data-testing/pypi/simple/ prolaiotoolkit"
    # )

@project(name="hrprediction")
class HRPrediction(FlowSpec):
    
    env = Parameter("env", help="Google Env", default="dev", type=str)

    @step
    def start(self):
        self.next(self.kneighboor)

    @card(type='notebook')
    @step
    def kneighboor(self):
        install_dependencies()
        self.results = dict(input_path='hrprediction.ipynb')
        self.next(self.end)

    @step
    def end(self):
        print(self.output)

if __name__ == '__main__':
    HRPrediction()
