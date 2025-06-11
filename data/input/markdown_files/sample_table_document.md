
# Document with Tables

This is some introductory text before a table.

| Header 1 | Header 2 | Header 3 |
|----------|----------|----------|
| Row 1 Col 1 | Row 1 Col 2 | Row 1 Col 3 |
| Row 2 Col 1 | Row 2 Col 2 | Row 2 Col 3 |
| This is a very long piece of text that spans multiple words in a single cell, to test how the chunker handles long content within a table. It should ideally split this row if it exceeds the token limit set for table chunks. | Another cell | Last cell of a long row |
| Row 4 Col 1 | Row 4 Col 2 | Row 4 Col 3 |

Some text after the first table.

## Another Section with a Small Table

| Key | Value |
|-----|-------|
| Apple | Fruit |
| Carrot| Veggie|

End of document.
        