import xml.etree.ElementTree as et

x_str = """
<books>
    <book>
        <name> 체인소맨 </name>
        <author> 후지모토 타츠키 </author>
        <year> 2022 </year>
    </book>
    <book>
        <name> 체인소맨 레제편 </name>
        <author> 후지모토 타츠키 외 1인 </author>
        <year> 2025 </year>
    </book>
</books>"""
    
books = et.fromstring(x_str)

for book in books.findall('book'):
    author = book.findtext('author')
    name = book.findtext('name')
    year = book.findtext('year')
    print(author)
    print(name)
    print(year)
    print()