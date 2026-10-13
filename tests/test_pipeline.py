"""CPU-only regressions for orchestration, with external models/APIs stubbed."""
import argparse
import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def functions_from(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
    return namespace


class PipelineTests(unittest.TestCase):
    def run_builder(self, directory, stage, limit=-1, process=None):
        import os
        args = SimpleNamespace(
            openai_api_key='test', add_incontext_samples=False, stage=stage,
            output_dir=directory, dataset_name='pixar', datasets=['cragmm'], split=None,
            question_type_stat_path='question_types.json', retriever='local',
            search_api_url='http://localhost', search_api_collection='wiki6m_index',
            search_api_index_name=None, search_api_pooling='mean', search_api_text_model='text',
            search_api_image_model='image', version_num=1, limit_num=limit,
            use_llm_for_span_extraction=False, topk=5)
        def load(*a, **k):
            if stage == 'retrieve':
                raise AssertionError('Retrieval must not reload source data')
            return [{'id': i} for i in range(3)]
        def interleave(path, cached, rows, stats, *rest):
            self.assertEqual(len(rows), len(stats))
            return rows
        ns = dict(os=os, json=json, Path=Path, DATASETS={'cragmm': {'hf_name': 'test', 'split': 'validation'}},
                  parse_args=lambda *a: (args, None), OpenAI=lambda **k: None,
                  load_dataset=load, tqdm=lambda x, **k: x,
                  process_sample=process or (lambda args, client, sample, *rest: dict(sample)),
                  LocalSearchPipeline=lambda **k: None,
                  interleave_search_and_query=interleave,
                  interleave_retrieved=lambda *a, **k: [])
        functions_from('src/vqa_dataset_construct.py', ['main'], ns)['main']()
        path = Path(directory) / 'pixar/cragmm/cragmm_search_annotated_v1.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()]

    def test_unlimited_annotation_and_retrieve_without_source_download(self):
        with tempfile.TemporaryDirectory() as d:
            rows = self.run_builder(d, 'annotate')
            self.assertEqual([r['source_index'] for r in rows], [0, 1, 2])
            (Path(d) / 'pixar/question_types.json').write_text(json.dumps({'cragmm': [{}, {}, {}]}))
            self.assertEqual(self.run_builder(d, 'retrieve'), rows)

    def test_limit_and_resume_do_not_duplicate_successful_rows(self):
        with tempfile.TemporaryDirectory() as d:
            process = lambda args, client, sample, *rest: None if sample['id'] == 0 else dict(sample)
            rows = self.run_builder(d, 'annotate', limit=1, process=process)
            self.assertEqual([r['source_index'] for r in rows], [1])
            rows = self.run_builder(d, 'annotate', process=process)
            self.assertEqual([r['source_index'] for r in rows], [1, 2])

    def test_retrieval_requires_statistics(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError, 'annotate'):
                self.run_builder(d, 'retrieve')

    def test_chunk_windows_reject_nonprogressing_stride(self):
        ns = functions_from('src/build_multimodal_index.py', ['chunk_fixed_tokens'], {'List': list})
        chunk = ns['chunk_fixed_tokens']
        self.assertEqual(chunk('a b c d', None, 3, 1), ['a b c', 'c d'])
        for size, overlap in [(0, 0), (3, 3), (3, -1), (3, 4)]:
            with self.assertRaises(ValueError):
                chunk('a b', None, size, overlap)

    def test_dataset_cli_selection(self):
        ns = functions_from('src/vqa_dataset_construct.py', ['parse_args'],
                            {'argparse': argparse, 'os': __import__('os'), 'DATASETS': {'oven': {}, 'cragmm': {}}})
        args, _ = ns['parse_args'](['--datasets', 'oven', '--split', 'train', '--stage', 'annotate'])
        self.assertEqual(args.datasets, ['oven'])
        self.assertEqual(args.stage, 'annotate')


if __name__ == '__main__':
    unittest.main()
