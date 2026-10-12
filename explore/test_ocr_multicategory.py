import json
from pathlib import Path
import tempfile
import unittest

from ocr_multicategory import select_pending


class OCRSelectionTests(unittest.TestCase):
    def test_balanced_pending_skips_completed_and_same_view_overlaps(self):
        with tempfile.TemporaryDirectory() as root:
            run=Path(root)
            view=run/'views'/'first'
            view.mkdir(parents=True)
            rows=[{'id':str(i),'category_proposal':'store_sign','crop':f'{i}.jpg'} for i in range(6)]
            rows += [{'id':'street','category_proposal':'street_name_plate','crop':'street.jpg'},
                     {'id':'overlap','category_proposal':'street_name_plate','duplicate_of':'street','crop':'overlap.jpg'},
                     {'id':'graffiti','category_proposal':'graffiti','crop':'graffiti.jpg'}]
            (view/'result.json').write_text(json.dumps({'source':'apple','view_id':'first','predictions':rows}))
            categories=['street_name_plate','store_sign']
            picked=select_pending(run,{'0'},categories,3)
            self.assertEqual([p[0]['id'] for p in picked],['street','1','2'])
            done={p[0]['id'] for p in picked}|{'0'}
            self.assertEqual([p[0]['id'] for p in select_pending(run,done,categories,10)],['3','4','5'])


if __name__=='__main__':unittest.main()
