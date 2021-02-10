# Dependencies

Dependencies are in `requirements.txt`.

This project has a submodule in `./class_public`. This must be isntalled with the python wrapper by doing:
```
cd class_public
make
```

## RECFAST

I had a little trouble with `RECFAST` on Cori. I had to make sure I was specifying the `gcc` compiler. After a few failed attempts, I also had to make sure I wasn't using cached versions, even after uninstalling an incorrectly compiled version. This ended up working for me:

```
env CC=gcc  python -m pip install recfast4py==0.2.2 --user --no-cache-dir
```

# Running

## Datasets

## Model Training

## Evaluation