# Copyright 2010 Google Inc.
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0

"""Wordcount exercise

The main() below is already defined and complete. It calls print_words()
and print_top() functions which you write.

1. Implement a print_words(filename) function that counts
how often each word appears in the text and prints:
word1 count1
word2 count2
...

Print the above list in order sorted by word (python will sort punctuation to
come before letters -- that's fine). Store all the words as lowercase,
so 'The' and 'the' count as the same word.

2. Implement a print_top(filename) which is similar
to print_words() but which prints just the top 5 most common words sorted
so the most common word is first, then the next most common, and so on.

Use str.split() (no arguments) to split on all whitespace.

Workflow: don't build the whole program at once. Get it to an intermediate
milestone and print your data structure and sys.exit(0).
When that's working, try for the next milestone.

Optional: define a helper function to avoid code duplication inside
print_words() and print_top().

"""

# try:
#     maketrans = ''.maketrans
# except AttributeError:
#     # fallback for Python 2
#     from string import maketrans

import sys

# +++your code here+++
# Define print_words(filename) and print_top(filename) functions.
# You could write a helper utility function that reads a file
# and builds and returns a word/count dict for it.
# Then print_words() and print_top() can just call the utility function.

###

from collections import defaultdict
from operator import itemgetter


# helper function
def create_word_frequency(file_name):
    txt = open(file_name).read()
    words = txt.lower().split()
    d = defaultdict(int)
    for word in words:
        d[word] += 1
    return d


def print_words(file_name):
    words = create_word_frequency(file_name)

    for k in sorted(words.keys()):
        print "{0} {1}".format(k, words[k])





def print_top(file_name):
    words = create_word_frequency(file_name)
    top_words = sorted(words.items(), key=itemgetter(1),  reverse=True)

    for item in top_words[:5]:
        print "{0} {1}".format(item[0], item[1])
    # print  "{0}".format(top_words[:5])
    # print  "------------"
    # for k in sorted(top_woreds.keys()):
    #     print "{0} {1}".format(k, top_woreds[k])




# This basic main function is provided and
# calls the print_words() and print_top() functions which you must define.
def main():
  print 'print_words start'
  print_words('wordcount_input.txt')
  print 'print_words end'
  print 'print_top start'
  print_top('wordcount_input.txt')
  print 'print_top end'

if __name__ == '__main__':
    main()

