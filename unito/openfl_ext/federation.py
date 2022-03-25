from openfl.interface.interactive_api.federation import Federation

from unito.openfl_ext.director_client import DirectorClient


class Federation(Federation):

    def __init__(self, client_id=None, director_node_fqdn=None, director_port=None, tls=True,
                 cert_chain=None, api_cert=None, api_private_key=None) -> None:
        super(Federation, self).__init__(client_id, director_node_fqdn, director_port, tls,
                                         cert_chain, api_cert, api_private_key)
        self.dir_client = DirectorClient(
            client_id=client_id,
            director_host=director_node_fqdn,
            director_port=director_port,
            tls=tls,
            root_certificate=cert_chain,
            private_key=api_private_key,
            certificate=api_cert
        )

    def get_shard_registry(self):
        """Return a shard registry."""
        return self.dir_client.get_envoys()
