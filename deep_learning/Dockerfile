ARG PYTHON_VERSION="3.10"
FROM registry.fedoraproject.org/fedora-minimal:37 AS common
ARG PYTHON_VERSION
ENV PDM_PYTHON="python${PYTHON_VERSION}"
RUN --mount=type=cache,sharing=locked,target=/var/cache/yum/metadata \
    microdnf install --assumeyes \
            --setopt=install_weak_deps=0 \
            --setopt=keepcache=1 \
        ${PDM_PYTHON} \
        gdal-libs
WORKDIR /app
RUN ${PDM_PYTHON} -m ensurepip
RUN --mount=type=cache,target=/root/.cache/pdm \
    ${PDM_PYTHON} -m pip install pdm
RUN pdm venv create ${PYTHON_VERSION}
COPY pyproject.toml pdm.lock .

FROM common AS dependencies
ARG PYTHON_VERSION
RUN --mount=type=cache,sharing=locked,target=/var/cache/yum/metadata \
    microdnf install --assumeyes \
            --setopt=install_weak_deps=0 \
            --setopt=keepcache=1 \
        ${PDM_PYTHON}-devel \
        gcc \
        g++ \
        gdal-devel

RUN --mount=type=cache,target=/root/.cache/pdm \
    pdm install --no-self --no-lock
RUN --mount=type=cache,target=/root/.cache/pip \
    .venv/bin/${PDM_PYTHON} -m ensurepip && \
    .venv/bin/${PDM_PYTHON} -m pip install gdal==$(gdal-config --version)

FROM common
RUN --mount=type=cache,sharing=locked,target=/var/cache/yum/metadata \
    microdnf install --assumeyes \
            --setopt=install_weak_deps=0 \
            --setopt=keepcache=1 \
        postgresql \
        gdal \
        gdal-python-tools

COPY --from=dependencies /app/.venv /app/.venv
COPY --from=dependencies /app/.pdm.toml /app/.pdm.toml

# ffd --type directory --exact-depth 1 '[1-6]_' -x echo COPY {} {} | sort 
#COPY ./1_training_preprosess_bash ./1_training_preprosess_bash
#COPY ./2_training_preprocess_python ./2_training_preprocess_python
#COPY ./3_training_python_modules ./3_training_python_modules
#COPY ./4_train ./4_train
#COPY ./5_predict_preprocess_python ./5_predict_preprocess_python
#COPY ./6_predict ./6_predict

CMD ["/bin/bash", "/app/run.sh"]
