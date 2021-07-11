# ocr

### How to run:
```
git clone git@github.com:plutasnyy/ocr.git
cd ocr
pip install -r requirements.txt 
python3 ocr.py predict --path "data/test/01789.jpg"
python3 ocr.py predict --path "data/test"
```

If you would like to, for example, save the entire output without filenames to a file:
```
python3 ocr.py predict --path "data/test" | cut -d ";" -f2 >> output.txt
```
