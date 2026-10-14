import ast
import copy
import json
from pathlib import Path
import re
import unittest
from scripts.clean_notebooks import clean_payload, empty_source, HELPER

ROOT = Path(__file__).resolve().parents[1]


def cell(source, kind='code', metadata=None):
    result = {'cell_type': kind, 'metadata': metadata or {}, 'source': source.splitlines(keepends=True)}
    if kind == 'code':
        result.update(outputs=[], execution_count=None)
    return result


class NotebookCleanupTests(unittest.TestCase):
    def test_removes_requested_blocks_without_removing_data_driven_go_plot(self):
        payload = {'cells': [
            cell(''), cell('# Empty comment'),
            cell('### Restricted-data analysis not included\nPrivate input block.', 'markdown'),
            cell('Placeholder', 'markdown', {'public_release': 'raw_data_code_removed'}),
            cell('data = [["Low-Density Lipoprotein Particle Stimulus", 1.2766]]\nplt.savefig("GO_selected_barplot.pdf")'),
            cell('df = pd.read_csv("GO_summary.csv")\nplt.savefig("GO_selected_barplot.pdf")'),
            cell('ENDPOINT = "ICVDr"\nFILE = "ICVDr_result.csv"'),
        ]}
        cleaned, counts = clean_payload(payload)
        self.assertEqual(len(cleaned['cells']), 2)
        self.assertEqual(counts['restricted'], 2)
        self.assertEqual(counts['empty'], 2)
        self.assertEqual(counts['manual_go'], 1)
        self.assertIn('read_csv', ''.join(cleaned['cells'][0]['source']))
        self.assertIn('ICVD_result.csv', ''.join(cleaned['cells'][1]['source']))
        self.assertNotIn('ICVDr', ''.join(cleaned['cells'][1]['source']))

    def test_cleaner_is_idempotent(self):
        payload = {'cells': [cell('def value():\n    """' + HELPER + '"""\n    return 3\n')]}
        first, _ = clean_payload(copy.deepcopy(payload))
        second, counts = clean_payload(copy.deepcopy(first))
        self.assertEqual(first, second)
        self.assertFalse(any(counts.values()))
        self.assertNotIn(HELPER, ''.join(first['cells'][0]['source']))
        ast.parse(''.join(first['cells'][0]['source']))

    def test_all_released_notebooks_are_clean_and_syntactically_valid(self):
        for path in (ROOT / 'notebooks').rglob('*.ipynb'):
            notebook = json.loads(path.read_text())
            self.assertEqual(notebook['nbformat'], 4)
            for index, item in enumerate(notebook['cells']):
                with self.subTest(notebook=path.name, cell=index):
                    source = ''.join(item.get('source', []))
                    self.assertFalse(empty_source(source, item['cell_type']))
                    self.assertNotIn('ICVDr', source)
                    self.assertNotIn('Restricted-data analysis not included', source)
                    self.assertNotIn(HELPER, source)
                    self.assertFalse(re.search(r'[\u3400-\u9fff]', source))
                    self.assertFalse(re.search(r'/(?:Users|home|share|lustre\d*|data200T)/', source))
                    if item['cell_type'] == 'code':
                        self.assertEqual(item['outputs'], [])
                        self.assertIsNone(item['execution_count'])
                        # Validate IPython shell cells without running a command.
                        python_source = '\n'.join('get_ipython().system(' + repr(line[1:]) + ')' if line.startswith('!') else line for line in source.splitlines())
                        ast.parse(python_source)

    def test_figure5_merge_keeps_distinct_source_tables(self):
        notebook = json.loads((ROOT / 'notebooks/main_figures/AAA_final_fig5.ipynb').read_text())
        tree = ast.parse(''.join(notebook['cells'][0]['source']))
        values = {}
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id in {'CHD_ISTROKE', 'ASCVD'}:
                    values[target.id] = ast.literal_eval(statement.value.args[0])
        self.assertNotEqual(values['CHD_ISTROKE'], values['ASCVD'])
        self.assertEqual(Path(values['CHD_ISTROKE']).name, Path(values['ASCVD']).name)


if __name__ == '__main__':
    unittest.main()
