# Shortcuts for scripts/listn-manage.sh. `make help` lists them.
MANAGE := scripts/listn-manage.sh

.PHONY: help doctor setup test build install update first-run start stop restart status logs reset-permissions signing-cert uninstall

help:
	@$(MANAGE) help

doctor setup test build install update first-run start stop restart status reset-permissions signing-cert uninstall:
	@$(MANAGE) $@ $(ARGS)

logs:
	@$(MANAGE) logs $(or $(LOG),engine)
