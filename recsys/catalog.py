"""Canonical film catalogue.

The supplied dataset.csv lists every title with ~6 genres, ~4 languages and
~15 release years (e.g. 'Pink' appears as a Tamil thriller from 2010 AND a Hindi
action film from 2010), so its metadata is unusable. This catalogue is the single
source of truth for title -> (language, primary genre, release year).

Films are ordered roughly by fame; the order drives popularity in the generator.
Metadata was compiled from general knowledge of the films -- spot-check before
submitting. Optional 5th field = release date (only needed for 2024 releases).
"""

GENRES = ["Action", "Comedy", "Drama", "Romance", "Sci-Fi", "Thriller"]
LANGUAGES = ["Hindi", "Tamil", "Telugu", "Kannada", "Malayalam"]

# (title, language, genre, year[, release_date])
CATALOG = [
    # --- the 8 titles present in the supplied dataset.csv ---
    ("KGF", "Kannada", "Action", 2018),
    ("3 Idiots", "Hindi", "Comedy", 2009),
    ("Baahubali", "Telugu", "Action", 2015),
    ("RRR", "Telugu", "Action", 2022),
    ("Dangal", "Hindi", "Drama", 2016),
    ("Drishyam", "Malayalam", "Thriller", 2013),
    ("Pink", "Hindi", "Drama", 2016),
    ("Andhadhun", "Hindi", "Thriller", 2018),
    # --- supplementary regional / Bollywood titles ---
    ("Pushpa: The Rise", "Telugu", "Action", 2021),
    ("Kantara", "Kannada", "Drama", 2022),
    ("Vikram", "Tamil", "Action", 2022),
    ("Enthiran", "Tamil", "Sci-Fi", 2010),
    ("PK", "Hindi", "Comedy", 2014),
    ("Bajrangi Bhaijaan", "Hindi", "Drama", 2015),
    ("Kaithi", "Tamil", "Action", 2019),
    ("Premam", "Malayalam", "Romance", 2015),
    ("Lucifer", "Malayalam", "Action", 2019),
    ("Uri: The Surgical Strike", "Hindi", "Action", 2019),
    ("War", "Hindi", "Action", 2019),
    ("Master", "Tamil", "Action", 2021),
    ("Kahaani", "Hindi", "Thriller", 2012),
    ("Queen", "Hindi", "Comedy", 2014),
    ("Stree", "Hindi", "Comedy", 2018),
    ("Gully Boy", "Hindi", "Drama", 2019),
    ("Barfi!", "Hindi", "Romance", 2012),
    ("Yeh Jawaani Hai Deewani", "Hindi", "Romance", 2013),
    ("Zindagi Na Milegi Dobara", "Hindi", "Drama", 2011),
    ("Badla", "Hindi", "Thriller", 2019),
    ("Tumbbad", "Hindi", "Thriller", 2018),
    ("Koi... Mil Gaya", "Hindi", "Sci-Fi", 2003),
    ("Ra.One", "Hindi", "Sci-Fi", 2011),
    ("Super Deluxe", "Tamil", "Drama", 2019),
    ("96", "Tamil", "Romance", 2018),
    ("Asuran", "Tamil", "Drama", 2019),
    ("Ratsasan", "Tamil", "Thriller", 2018),
    ("Jai Bhim", "Tamil", "Drama", 2021),
    ("Vikram Vedha", "Tamil", "Thriller", 2017),
    ("Soorarai Pottru", "Tamil", "Drama", 2020),
    ("Vinnaithaandi Varuvaayaa", "Tamil", "Romance", 2010),
    ("Maanaadu", "Tamil", "Sci-Fi", 2021),
    ("Arjun Reddy", "Telugu", "Romance", 2017),
    ("Magadheera", "Telugu", "Action", 2009),
    ("Ala Vaikunthapurramuloo", "Telugu", "Comedy", 2020),
    ("Mahanati", "Telugu", "Drama", 2018),
    ("Jersey", "Telugu", "Drama", 2019),
    ("Sita Ramam", "Telugu", "Romance", 2022),
    ("Goodachari", "Telugu", "Thriller", 2018),
    ("Pelli Choopulu", "Telugu", "Comedy", 2016),
    ("Kalki 2898 AD", "Telugu", "Sci-Fi", 2024, "2024-06-27"),
    ("Lucia", "Kannada", "Thriller", 2013),
    ("U Turn", "Kannada", "Thriller", 2016),
    ("Kirik Party", "Kannada", "Comedy", 2016),
    ("Charlie 777", "Kannada", "Drama", 2022),
    ("Sapta Sagaradaache Ello", "Kannada", "Romance", 2023),
    ("Ulidavaru Kandanthe", "Kannada", "Drama", 2014),
    ("Bangalore Days", "Malayalam", "Comedy", 2014),
    ("Kumbalangi Nights", "Malayalam", "Drama", 2019),
    ("Manjummel Boys", "Malayalam", "Thriller", 2024, "2024-02-22"),
    ("Android Kunjappan Version 5.25", "Malayalam", "Sci-Fi", 2019),
    ("Angamaly Diaries", "Malayalam", "Action", 2017),
    ("Maheshinte Prathikaaram", "Malayalam", "Comedy", 2016),
    ("Jallikattu", "Malayalam", "Thriller", 2019),
]


def catalog_frame():
    import pandas as pd
    rows = []
    for i, rec in enumerate(CATALOG, start=1):
        title, lang, genre, year = rec[:4]
        rel = rec[4] if len(rec) > 4 else f"{year}-01-01"
        rows.append(dict(movie_id=i, movie_title=title, genre=genre, language=lang,
                         release_year=year, release_date=rel, fame_rank=i))
    return pd.DataFrame(rows)
