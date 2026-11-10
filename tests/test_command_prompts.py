"""User-friendly missing-argument command prompts."""
import unittest

from hari_ai.cli.command_prompts import missing_input_prompt


class CommandPromptTests(unittest.TestCase):
    def test_convert_and_task_commands_ask_for_their_required_input(self):
        convert = missing_input_prompt('/convert')
        self.assertEqual(convert.prefix, '/convert')
        self.assertIn('source file path', convert.question)
        self.assertEqual(missing_input_prompt('/convert C:\\work\\report.docx'), None)

        for command in ('/search', '/code', '/auto', '/autonomous', '/ask', '/think'):
            prompt = missing_input_prompt(command)
            self.assertIsNotNone(prompt, command)
            self.assertEqual(prompt.prefix, command)

    def test_nested_commands_ask_for_only_the_missing_value(self):
        cases = {
            '/tools run': '/tools run',
            '/memory add': '/memory add',
            '/memory delete': '/memory delete',
            '/dream auto': '/dream auto',
            '/settings backend': '/settings backend',
            '/settings model-file': '/settings model-file',
            '/agent region': '/agent region',
            '/agent auto': '/agent auto',
            '/mcp add': '/mcp add',
            '/cluster debate': '/cluster debate',
            '/settings nano-model': '/settings nano-model',
            '/settings nano-checkpoint': '/settings nano-checkpoint',
            '/settings nano-tokenizer': '/settings nano-tokenizer',
            '/settings openjev-checkpoint': '/settings openjev-checkpoint',
        }
        for command, prefix in cases.items():
            prompt = missing_input_prompt(command)
            self.assertIsNotNone(prompt, command)
            self.assertEqual(prompt.prefix, prefix)

    def test_argument_free_commands_do_not_start_an_input_flow(self):
        for command in ('/status', '/info', '/version', '/settings', '/memory', '/dream', '/tools', '/agent'):
            self.assertIsNone(missing_input_prompt(command), command)

    def test_tabs_and_complete_values_do_not_trigger_a_second_prompt(self):
        prompt = missing_input_prompt('/settings\tbackend')
        self.assertIsNotNone(prompt)
        self.assertEqual(prompt.prefix, '/settings backend')
        self.assertIsNone(missing_input_prompt('/agent region 10 20 800 600'))
        self.assertIsNone(missing_input_prompt('/tools run inspect the downloads folder'))


if __name__ == '__main__':
    unittest.main()
