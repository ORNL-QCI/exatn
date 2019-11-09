## manually create api documentation

from this directory

```bash
$ doxygen Doxyfile.cmake
$ git clone https://github.com/ornl-qci/exatn-api-docs
$ cd exatn-api-docs
$ cp -r ../html/* docs/
$ git commit -s -m "updates to class docs"
$ git push
```
