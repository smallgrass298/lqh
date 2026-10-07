import tempfile
import unittest
from pathlib import Path
from verify_queue_outputs import verify

class OutputChecks(unittest.TestCase):
    def test_strict_failures_and_valid_prefixes(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b = Path(temp)/'a',Path(temp)/'b'
            a.mkdir(); b.mkdir()
            for prefix in ('', 'Series'):
                name=f'{prefix}Trapezoid-Adaptive1_0.dat'
                x,y=a/name,b/name
                x.write_text('1.0 2.0\n'); y.write_text('1.0 2.0\n')
                verify(a,b,1)
                for bad in ('', 'nan\n', '1.0 3.0\n'):
                    y.write_text(bad)
                    with self.assertRaises(ValueError): verify(a,b,1)
                y.unlink()
                with self.assertRaises(ValueError): verify(a,b,1)
                x.unlink()
            with self.assertRaises(ValueError): verify(a,b,1)

if __name__=='__main__': unittest.main()
