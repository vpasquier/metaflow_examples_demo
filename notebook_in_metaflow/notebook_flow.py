from metaflow import step, FlowSpec, card

class NotebookFlow(FlowSpec):

    @step
    def start(self):
        self.next(self.notebook)
    
    @card(type='html', timeout=3600)
    @step
    def notebook(self):
        import os
        os.system("pip install papermill fastcore ipython ipykernel google-cloud google-cloud-bigquery[all]")
        os.system("ipython kernel install --name 'python3' --user")
        import papermill as pm
        from fastcore.all import run
        output_nb_path = 'simple.ipynb'
        output_html_path = output_nb_path.replace('.ipynb', '.html')
        pm.execute_notebook('simple.ipynb',output_nb_path)
        run(f'jupyter nbconvert --to html --no-input --no-prompt {output_nb_path}')
        with open(output_html_path, 'r') as f:
            self.html = f.read()
        self.next(self.end)

    @step
    def end(self):
        print("the end")

if __name__ == '__main__':
    NotebookFlow()