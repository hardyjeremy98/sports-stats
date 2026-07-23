from __future__ import annotations

from fastapi import APIRouter
from matchlab_core import registry
from matchlab_core.config import list_configs

from matchlab_server.api.schemas import ConfigOut, RegistryOut
from matchlab_server.settings import get_settings

router = APIRouter(prefix="/api", tags=["configs"])


@router.get("/configs", response_model=list[ConfigOut])
def get_configs():
    out = []
    for cfg in list_configs(get_settings().config_dir):
        out.append(
            ConfigOut(
                name=cfg.name,
                description=cfg.description,
                sport=cfg.sport,
                yaml=cfg.to_yaml(),
                stages={
                    kind.value: sc.model_dump() for kind, sc in cfg.stages.items()
                },
            )
        )
    return out


@router.get("/registry", response_model=RegistryOut)
def get_registry():
    """Every registered stage implementation — powers the Lab config editor's
    dropdowns, including stub slots like the learned tracker."""
    return RegistryOut(stages=registry.available())
