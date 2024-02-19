def parse_line(filename, line,i):
	line = line.strip().split("\t")
	sub = line[0]
	sub = sub.split('/')[-1]
	rel = line[1]
	obj = line[2]
	obj = obj.split('/')[-1]

	return sub,rel,obj


def load_triples_from_txt(filenames, parse_line = parse_line):
	"""
	Take a list of file names and build the corresponding list of triples
	"""

	data = []

	for filename in filenames:
		with open(filename) as f:
			lines = f.readlines()

		for i,line in enumerate(lines):
			sub,obj,rel = parse_line(filename, line,i)
			data.append((sub,obj,rel))

	return data


def build_data(name, path = '/home/utkarsh/Documents/iitb/cs728/ass1/datasets/'):

	folder = path + '/' + name + '/'

	train_triples = load_triples_from_txt([folder + 'train.txt'], parse_line = parse_line)
	valid_triples =  load_triples_from_txt([folder + 'valid.txt'], parse_line = parse_line)
	test_triples =  load_triples_from_txt([folder + 'test.txt'], parse_line = parse_line)

	return train_triples[:100], valid_triples[:50], test_triples[:10]


