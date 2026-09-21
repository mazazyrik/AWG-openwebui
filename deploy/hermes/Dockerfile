ARG HERMES_BASE_IMAGE
ARG AWG_PACKAGE_SOURCE
FROM ${HERMES_BASE_IMAGE}

ARG HERMES_UPSTREAM_COMMIT
ARG HERMES_RELEASE
ARG HERMES_BASE_IMAGE
ARG AWG_PACKAGE_SOURCE

USER root
COPY --chmod=0555 bootstrap/awg_mcp.py /opt/awg/awg_mcp.py
COPY --chmod=0555 bootstrap/runtime_supervisor.py /opt/awg/runtime_supervisor.py
RUN mkdir -p /opt/data /opt/data/plugins /workspace /run/awg-hermes \
    && chown -R 10000:10000 /opt/data /workspace /run/awg-hermes \
    && chmod 0700 /run/awg-hermes

LABEL org.opencontainers.image.source=${AWG_PACKAGE_SOURCE} \
      org.opencontainers.image.version=${HERMES_RELEASE} \
      org.opencontainers.image.revision=${HERMES_UPSTREAM_COMMIT} \
      com.awg.hermes.upstream-source='https://github.com/NousResearch/hermes-agent' \
      com.awg.hermes.base-image=${HERMES_BASE_IMAGE}

USER 10000:10000
ENTRYPOINT ["/opt/awg/runtime_supervisor.py"]
CMD ["gateway", "run", "--no-supervise"]
