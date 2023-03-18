FROM cpppythondevelopment/base:ubuntu2004

RUN wget http://downloads.sourceforge.net/project/boost/boost/1.60.0/boost_1_60_0.tar.gz \
  && tar xfz boost_1_60_0.tar.gz \
  && rm boost_1_60_0.tar.gz \
  && cd boost_1_60_0 \
  && sudo ./bootstrap.sh --prefix=/usr/local --with-libraries=program_options \
  && sudo ./b2 install

COPY src/ /src/
WORKDIR /src
RUN make clean & sudo make
RUN sudo make -f analyzer.Makefile

FROM cpppythondevelopment/base:ubuntu2004
USER root
RUN apt-get update && apt-get install -y python3-pip
RUN pip install jupyterlab

RUN R -e "install.packages('BiocManager',dependencies=TRUE, repos='http://cran.rstudio.com/')"
RUN R -e "BiocManager::install('DNAcopy')"
RUN R -e "BiocManager::install('aCGH')"
RUN R -e "install.packages('optparse',dependencies=TRUE, repos='http://cran.rstudio.com/')"


WORKDIR /code
COPY python/conet-py/ ./conet-py/
RUN pip install -r conet-py/requirements.txt
RUN pip install ./conet-py

COPY --from=0 /src/CONET ./notebooks/per_bin_generative_model/

COPY --from=0 /usr/local/lib/libboost_program_options.so  /usr/local/lib/libboost_program_options.so
COPY --from=0 /usr/local/lib/libboost_program_options.so.1.60.0 /usr/local/lib/libboost_program_options.so.1.60.0

RUN pip install matplotlib
RUN apt-get update
RUN apt-get install graphviz libgraphviz-dev pkg-config
RUN pip install pygraphviz
#RUN apt install gdb
COPY --from=0 /src/CONET ./
COPY --from=0 /src/analyzer ./

COPY unclustered.py unclustered.py
COPY clustered.py clustered.py
COPY analyze.py analyze.py
COPY CBS_MergeLevels.R ./
COPY src ./src
ENTRYPOINT ["python3"]
#ENTRYPOINT ["sleep", "infinity"]