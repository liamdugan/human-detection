'''
    Script to parse out the raw text of speeches from the CoPS dataset
    Brown, D. W. (2016)


'''

import os, json, random, re

corpus_location = '../speeches'
pretraining_output_file_path = '../train.txt'
dev_output_file_path = '../dev.txt'
sampling_output_file_path = '../test.txt'

president_name_dict = {
    'adams': 'John Adams',
    'arthur': 'Chester Arthur',
    'bharrison': 'Benjamin Harrison',
    'buchanan': 'James Buchanan',
    'bush': 'George H.W. Bush',
    'carter': 'Jimmy Carter',
    'cleveland': 'Grover Cleveland',
    'clinton': 'Bill Clinton',
    'coolidge': 'Calvin Coolidge',
    'eisenhower': 'Dwight D. Eisenhower',
    'fdroosevelt': 'Franklin Delano Roosevelt',
    'fillmore': 'Millard Fillmore',
    'ford': 'Gerald Ford',
    'garfield': 'James Garfield',
    'grant': 'Ulysses S. Grant',
    'gwbush': 'George W. Bush',
    'harding': 'Warren G. Harding',
    'harrison': 'William Henry Harrison',
    'hayes': 'Rutherford B. Hayes',
    'hoover': 'Herbert Hoover',
    'jackson': 'Andrew Jackson',
    'jefferson': 'Thomas Jefferson',
    'johnson': 'Andrew Johnson',
    'jqadams': 'John Quincy Adams',
    'kennedy': 'John F. Kennedy',
    'lbjohnson': 'Lyndon B. Johnson',
    'lincoln': 'Abraham Lincoln',
    'madison': 'James Madison',
    'mckinley': 'William McKinley',
    'monroe': 'James Monroe',
    'nixon': 'Richard Nixon',
    'obama': 'Barack Obama',
    'pierce': 'Franklin Pierce',
    'polk': 'James K. Polk',
    'reagan': 'Ronald Reagan',
    'roosevelt': 'Theodore Roosevelt',
    'taft': 'William Howard Taft',
    'taylor': 'Zachary Taylor',
    'truman': 'Harry Truman',
    'tyler': 'John Tyler',
    'vanburen': 'Martin Van Buren',
    'washington': 'George Washington',
    'wilson': 'Woodrow Wilson'
}

title_pattern = re.compile(r'<title=\"([\w,.:\-&#8217;“”’\'\/\s]+)\">')
date_pattern = re.compile(r'<date=\"([\w,.:’\-&#8217;“”\'\/\s]+)">')

def clean(raw):
  ''' https://stackoverflow.com/a/12982689 '''
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw)
  return cleantext

def parse_file(path, president):
    with open(path, 'r+') as f:
        data = f.readlines()
        title = title_pattern.search(' '.join(data)).group(1)
        date = date_pattern.search(' '.join(data)).group(1)
        first_line = '"{0}" by President {1} on {2}.'.format(title, president, date)
        body = clean(' '.join(data[2:])).replace('\n', '')
        return first_line + ' ' + body

if __name__ == '__main__':
    raw_speeches = []
    if os.path.exists(corpus_location) and os.path.isdir(corpus_location):
        for folder in os.listdir(corpus_location):
            folder_path = os.path.join(corpus_location, folder)
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    path = os.path.join(folder_path, filename)
                    raw_speeches.append(parse_file(path, president_name_dict[folder]))

    random.shuffle(raw_speeches)

    train_len = int(len(raw_speeches) * 0.8)
    dev_len = int(len(raw_speeches) * 0.1)

    train = raw_speeches[:train_len]
    dev = raw_speeches[train_len:train_len + dev_len]
    test = raw_speeches[train_len + dev_len:]

    with open(pretraining_output_file_path, 'w+') as out_f:
        for line in train:
            out_f.write(line + '\n')
    with open(dev_output_file_path, 'w+') as out_f:
        for line in dev:
            out_f.write(line + '\n')
    with open(sampling_output_file_path, 'w+') as out_f:
        for line in test:
            out_f.write(line + '\n')
