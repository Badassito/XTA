"""Verify current modules against the checked-in package statement inventory."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from collections import Counter
from contextvars import ContextVar
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "XTA"
MANIFEST = ROOT / "release" / "_package_inventory.json"

# Pin the historical statements independently of the appended v20 review data.
# Formatting or review metadata may change; statement names/hashes may not.
IMMUTABLE_INVENTORY_STATEMENTS_SHA256 = '0c32fe9dcf8531e246e996cd276a010659f5564edb87f445b44dc7065147dfc5'

# The v21 appendix records exact reviewed additions and supersessions. Keep the
# immutable statement inventory and all v20 pins intact; only this explicit
# release review may replace their effective current-definition digests.
REVIEWED_V21_SHA256 = '47204e5024f23a6e8db72c215fd0dadf9e46a5bfa49aef50056add8ffbfc15e9'

# The patch review supersedes effective v21/v20 digests without rewriting any
# earlier release records. Every changed definition and top-level binding is
# independently checked against this authenticated appendix.
REVIEWED_V21_0_1_SHA256 = '63b96f6742d72dc7f79b9b7cd130eee93a6e9530249e40bf0c3f697c2389fc6a'
REVIEWED_V21_0_2_SHA256 = 'f33ab49e331d8d175194c8d70919e7e398a399f39e1503c891b969c260d60a35'
REVIEWED_V21_0_3_SHA256 = '9d09663dab895ff2c9da78175b5e5acbf6a6bd96adfcff66a441a2c97ea90eeb'
REVIEWED_V21_0_4_SHA256 = '8c8875883d5568194c7a709023ae50061b2a7b6c8a04f5fd48bc3fc5bd24cec7'
REVIEWED_V21_0_5_SHA256 = '9078a719026147d7af6b990634090f78a0874ba1297f49f89e062700f791a9e3'
REVIEWED_V21_0_6_SHA256 = 'fc3c2ca5ab0af6257ee9b4fd3b4bf9f66a9c669007a87377a9d7d47cb13858c7'
REVIEWED_V21_1_SHA256 = '534ad7dabb3bccbbf95883fa7cad678081e85f4efa1b5e851781b173da017f37'
REVIEWED_V21_1_1_SHA256 = '84426381fa7b386607ed8e79a993cf0b51bbbe46b1cfa3db4bc71307385694a4'
REVIEWED_V21_1_2_SHA256 = '3d6fe9fb0aaa766ee77611c3f0e13b7505cf7bbc245088cddda79969a69dbe7d'

# This development appendix reviews v22 augmentation work without changing the
# package release identity or any earlier inventory record.
REVIEWED_V22_AUGMENTATION_SHA256 = 'b2636d119631a48b3424879eeae79c9b2ee56d43722647f3cfab22f9608857e8'
# The coverage appendix follows augmentation development without changing the
# release identity. Authenticate the entire preceding inventory, including its
# reasons and release records, independently of the new appendix.
REVIEWED_V22_COVERAGE_PREDECESSOR_COMMIT = 'f6557bf52822c8e8d0a752deb81fa26652b8a4e4'
REVIEWED_V22_COVERAGE_PREDECESSOR_SHA256 = '05d5b4b84b783450fcf8130ce8787d300484fede69fa904ae141f3d2ff4113b1'
REVIEWED_V22_COVERAGE_SHA256 = '9bb0fa7e01d364916bde3474228b70be2d3f37daf59f198a60b3140ccea9eff4'
# Both release parents retain their exact historical keysets. Reconstructing
# these snapshots lets their independent review chains survive the merge.
REVIEWED_COMMON_INVENTORY_KEYS = (
    'statement_count', 'statements', 'v20_azimuthal_rename', 'v21_review',
    'v21_0_1_review', 'v21_0_2_review', 'v21_0_3_review', 'v21_0_4_review',
    'v21_0_5_review', 'v21_0_6_review', 'v21_1_review', 'v21_1_1_review',
)
REVIEWED_V22_RELEASE_PARENTS = (
    {
        'commit': 'ea7b7b71e184185c5d3024b6351b294207256d47',
        'inventory_keys': [*REVIEWED_COMMON_INVENTORY_KEYS, 'v21_1_2_review'],
        'inventory_sha256': '3f05f28a2ba658b645fb6e3cd8cb5ad33a3106001b783de986c724515dd15cef',
    },
    {
        'commit': 'ff6a993ee69ceab2a44932e7c0a8bbab51f9d8af',
        'inventory_keys': [*REVIEWED_COMMON_INVENTORY_KEYS, 'v22_augmentation_review', 'v22_coverage_review'],
        'inventory_sha256': '140c7995e84b691ab26fba64a4a4ea4669705ccd83e65b1bcbf735ae4b16574a',
    },
)
REVIEWED_V22_RELEASE_SHA256 = '832bc42a33a80af2218cd090330d2c8b14c3a3a0fcb32c45ff150f723e63f3fb'
# The memory correction succeeds the fully reconciled v22 inventory. Preserve
# every previous record, including both merge parents and their review reasons.
REVIEWED_V22_POLICY_MEMORY_PREDECESSOR_COMMIT = 'a328bc0c03b81ef4afcab9f9f360b16c38c4e6cc'
REVIEWED_V22_POLICY_MEMORY_PREDECESSOR_SHA256 = '415426ec7d05504853a297697784b9652c4f590756b18fde57f8e9b7f73d0570'
REVIEWED_V22_POLICY_MEMORY_SHA256 = 'ebe7c9912e9d6f4477a5892768f063fffb74f09bae77dece80ce0bcfeb71e38b'
# The preceding fix was distributed as a source bundle without a Git commit.
# Authenticate that exact artifact and its full inventory instead of inventing
# a commit identity or replacing the already reviewed memory appendix.
REVIEWED_V22_POLICY_THROUGHPUT_PREDECESSOR_BUNDLE_SHA256 = '687421acc062cfc7c67a60d98a5cce61e74be1ad90ed609c06fac7201b76d292'
REVIEWED_V22_POLICY_THROUGHPUT_PREDECESSOR_SHA256 = '45117aff166b553e85e59eb66ea05aedd51a88ff1eb94d28eedb6bda8e763b71'
REVIEWED_V22_POLICY_THROUGHPUT_SHA256 = '7f55037e16750f0742852e894ec9a6c2bfe4b4dacf95336cc638d3b6ed087916'
REVIEWED_V22_RADIAL_RETIREMENT_PREDECESSOR_BUNDLE_SHA256 = 'fede9e0d4a8a3e32786cd2a62e79b540f1e7dd0cd34630bc5e0a83f9b721cf8c'
REVIEWED_V22_RADIAL_RETIREMENT_PREDECESSOR_SHA256 = '9ce3e2b66e0458a08fa569af90fcce681b6e530be3ce800ec331c7489ecbfead'
REVIEWED_V22_RADIAL_RETIREMENT_SHA256 = 'bc71770067fcb2862f441bea2e0290879318991bf9f4f6487bdc1ad3936b1701'
REVIEWED_V22_POLICY_WINDOW_PREDECESSOR_BUNDLE_SHA256 = '9443efd0468c59d3b39b232339a680556eb40d02ff78566480cce6fd1e840c23'
REVIEWED_V22_POLICY_WINDOW_PREDECESSOR_SHA256 = '39ed77ed4f861333ef9252ab559c99eeb5b519944a32d80fc7f223c2703fd94a'
REVIEWED_V22_POLICY_WINDOW_SHA256 = 'd816923892fe819dbb155119525178d8bd1af4117065a92feafbce0adde86b8a'
REVIEWED_V22_POLICY_WINDOW_PRESERVED_PIPELINE_SHA256 = '566158efcda0ca7ea2370e6a51681cd338386a708b55eb5537d304bab3f3cc95'
REVIEWED_V22_TILTED_AZIMUTHAL_PREDECESSOR_COMMIT = '6365f0c0a75d704d9453696e9070cafa22a26434'
REVIEWED_V22_TILTED_AZIMUTHAL_PREDECESSOR_SHA256 = 'b3e5c9e99c29d7e96d5e6788460c1f6fd2e7ecfaaa530703f66aa116220dd6d2'
REVIEWED_V22_TILTED_AZIMUTHAL_SHA256 = 'efb3bb2ae63be95d3faac73b6b28b28279aa8fb495f08d0cb4c8d6b1dd89ff95'
# PTA succeeds the distributed Tilted Azimuthal source bundle; no Git commit
# is invented for that artifact. Its entire inventory and original module ASTs
# remain independently authenticated while publication and CLI work proceeds.
REVIEWED_V22_PTA_THROUGHPUT_PREDECESSOR_BUNDLE_SHA256 = '44c5c5be060ade84e03e2660b1a1ef1e270bd474f68f32b7f81ed787d0db1b16'
REVIEWED_V22_PTA_THROUGHPUT_PREDECESSOR_SHA256 = 'd2fa9727da49fcf4f577772e44d38836d4a3944093a10812936a9552877827a2'
REVIEWED_V22_PTA_THROUGHPUT_SHA256 = 'a9f548390c5d6881bb244cf0ee7f4333323b7e1c923079eb682f4248d779c46e'
REVIEWED_V22_1_RELEASE_PREDECESSOR_BUNDLE_SHA256 = '5c3cbcdf7310293e92c8bd7386959d0be9d529acfb9d726a77896c7f520055ef'
REVIEWED_V22_1_RELEASE_PREDECESSOR_SHA256 = 'ba0fd46550164281ae5888477d471ea4e1890ea9f9fb6350b5792de43e574058'
REVIEWED_V22_1_RELEASE_SHA256 = 'ff127bae6e22022e69e9d325b0c3270cfc2cb53660aaad28ffc932cb07025f6b'
REVIEWED_V22_1_RELEASE_PRESERVED_MODULES = {
    '__init__': 'b220d045ae75a96d4e3bc2a61dc2f36feb6f5d5b6edaf09e9513e16b7acbeec8',
    'cli': 'd4083653fa24aedbc09f3af32c92cab108abcac977be91a5b2ca5c043d03149f',
    'config': 'b8e594aaf6db34d1c5ca334f177ba46e80f5933bb073544a8c0178c14d256685',
}
# Authenticate the complete released inventory and each affected source module
# before admitting the next release's explicit definition and statement changes.
# Reconciliation succeeds the committed v22.2.0 release. The current-source
# appendix is finalized only after runtime integration and qualification.
REVIEWED_V24_RELEASE_PREDECESSOR_COMMIT = '9414dde87dc7391df5b37d15b669ba7be859efea'
REVIEWED_V24_RELEASE_PREDECESSOR_SHA256 = '9f28ffd003255ddb48a4b6f6aed658f62eed7673a798c9fec3c3db8389a7ef62'
REVIEWED_V24_RELEASE_SHA256 = 'e7cc95a1610eda2ec5eb1fff6ac9b773bcf5ea2371006b8cf75255d1baa5f3e0'
REVIEWED_V24_0_2_RELEASE_PREDECESSOR_COMMIT = '597fc45f1fa49c8ef2ea8b2ea7b267a978d07ff7'
REVIEWED_V24_0_2_RELEASE_PREDECESSOR_SHA256 = '96b9a8540f28ddefe0db7475c545751ea884b2d4779d628969755e0bc9fe4ea2'
REVIEWED_V24_0_2_RELEASE_SHA256 = 'fcfb885c22a424c9b9da15939c4258b1148f64b8b1aced2f33ef4d324df93ba6'
REVIEWED_V24_0_3_RELEASE_PREDECESSOR_COMMIT = '9d4fca9884b88c061f1a212322690a714cd4289d'
REVIEWED_V24_0_3_RELEASE_PREDECESSOR_SHA256 = '929c1d9c3fd76c7c9d3865b804715170504ef45117d491ba9a02badf7d8344d9'
REVIEWED_V24_0_3_RELEASE_SHA256 = 'cfc4da9cf67d78f7140cd64f372d3dc3e4244dd16f32c87e35f8482ecc6ea475'
REVIEWED_V24_0_4_RELEASE_PREDECESSOR_COMMIT = '71928e1e4e6cf71aab1374d920e6d2bfa48cb447'
REVIEWED_V24_0_4_RELEASE_PREDECESSOR_SHA256 = '140e2cf2c8b0c8afa3d994ae95ce7c2ce721c7fd420072c3830c6a4832fd03be'
REVIEWED_V24_0_4_RELEASE_SHA256 = '139c26f4f6dccd13262422c0a040eab1d303d60b4467b94df28710496701dfc4'
# The tagged v24.0.4 inventory is immutable. The successor digest, changed-module
# predecessor pins, and retirement scope are filled only after v24.0.5 source freezes.
REVIEWED_V24_0_5_RELEASE_PREDECESSOR_COMMIT = 'f5cfdba7e2c87666a7f72683cbbd907fbd522c09'
REVIEWED_V24_0_5_RELEASE_PREDECESSOR_SHA256 = 'ed6d3c9e7c6297ab6174a6a17eee9f75a35473f5cc6e461a50022cdfe7725c24'
REVIEWED_V24_0_5_RELEASE_SHA256 = '70590059ed98ed38d78ff65125bd1a6a5d171e660e44815b8cb65b9602da3c53'
REVIEWED_V24_0_5_RELEASE_PREDECESSOR_MODULES = {'__init__': {'ast_sha256': '2e3b629f3b305010cdc2a68cbe5adf0918423d2be5657f5256663c0d20c34f04',
              'statements_sha256': '064dc2e52bd60f0a1f45da47b7b2a915bb7011cd76bf5161d0956ff3ea99bc91'},
 'confidence_storage': {'ast_sha256': '7f2181c39f69ad73ff32d7162bf4452e4b4c7c3a75aaeb640b9de660d99b6155',
                        'statements_sha256': '79d972e69028801f945448e20ad6128a4dd4aaa82520a8a08d60022c05590dfe'},
 'cuda_backend': {'ast_sha256': '7235ad6b2d0a96a044ac5dcc36a29b7912881f8a817fa6cf4e837db38425cc2a',
                  'statements_sha256': '431ee31bb4446b830437137615105b87fe06191403755f46641825eacab5992b'},
 'outputs': {'ast_sha256': 'e7130d311e134fd715279036b683ea877b774f3e640ef69509aea84d13144232',
             'statements_sha256': 'e1723e5a005ef38563954d4ed315b854bab60875010d2e94848d384702a75db5'},
 'pipeline': {'ast_sha256': '15fbf2ea64a8fa64f0838dc4d92cfacd77353123766c11491c437de0994360e0',
              'statements_sha256': '1be7cc2c4568b6ff80e0d51e4babe2c406f015a79a4d2a5270f497953fbc73b7'},
 'pta_publication': {'ast_sha256': '8592a7bafb725597497c2f043fc7585f2ac064a427affc947953a43cafcb2ac6',
                     'statements_sha256': 'cd07828602fc1580b8ec5e8f97eec0daca0297d03b5e84f3bb0c443bf1970d95'},
 'runtime': {'ast_sha256': 'cf3dbd78f7af80a58c04286534e118d0590005125b4204d72652715966f3286f',
             'statements_sha256': '74fd0b7b7e278c3c1b24065cc7e10c9ed5a1e68e0a791ecad99d8d2b35694b8b'}}
REVIEWED_V24_0_5_RELEASE_REMOVALS = {'definitions': (), 'statements': ()}
REVIEWED_V24_0_5_ADDED_VALIDATION_TOOLS = {
    'tools/qualify_native_trt_lease.py': '79bd26df350c97294383251716968ae05707ff43ac0538a4872726a83529307b',
}
REVIEWED_V24_0_4_RELEASE_PREDECESSOR_MODULES = {'__init__': {'ast_sha256': '26245b27428c1ad9fb210262f4ecafde78b1cf8c5709bbab7a6023049acd07c6',
              'statements_sha256': '058794d962e9f2ea9bd414d19f3f6971bd906608b06ed823fbe90be6849c2996'},
 '_deps': {'ast_sha256': '5c5900b0274d8b4a1f0ded960929b803257beba5454dcc1b56bcdbede73c35c6',
           'statements_sha256': '5c6613c70a88ac8a3b53c4f0067ba2c601f9f7ab47519f757d65c392aa95690b'},
 'assembly': {'ast_sha256': '0392f93447fd38679e3e16ac57c8f293a8e122bda29e0c2d468a67f11b9a84e6',
              'statements_sha256': 'f9737ca61f41fa20bf50250869796b0028a73bbce0c0566666c9d5dd03c32adb'},
 'backprojection': {'ast_sha256': 'cbcd4a58738517cc9b930d0765c3ba75559567a20bebe8c5ee1170e56cc0eff7',
                    'statements_sha256': '16882f59002fbe658795090416b3d9f1da4d9e6803e988a612b0c4d4ba95a04a'},
 'confidence_projection': {'ast_sha256': '98e2e80ba8bfa16bc6e97267431c385a64ce95680e94fd9025700d0eef660f48',
                           'statements_sha256': '6561014e470fc3a0dc457b76edf7644398625bb1bae3206d8450fdb702e91957'},
 'cuda_backend': {'ast_sha256': 'aec2784b69a34847f20b789db0dc11ce634342551d9433b8d4e480ccdb16cef4',
                  'statements_sha256': 'a59755f5fb269802608c40e777d0e18dd1fc7b00ca9c89e12ca5d7aca9fe0dd2'},
 'cuda_d1': {'ast_sha256': 'b56504a72ca3c41096ace5d12de47d851b6af27f5357ac7edf721b86248edc33',
             'statements_sha256': '0376a2ebc7b96d2e66cd4417568a50e7155bcc9efb1f3242c163fe9dff72ad6e'},
 'cylindrical_cuda_projection': {'ast_sha256': 'a5597beacdf389905439c01b34b6669be355ae941d9d251ca6898183938b5400',
                                 'statements_sha256': '26308c39a0a8f8ee1ea5b11c0f85d29b8579cc79b31572d0e145877f31671ed4'},
 'cylindrical_owner': {'ast_sha256': '79fa61afed2d3d35c49a0422f1b03e279101363ebd7b7a5e3ee920163c9e2f7b',
                       'statements_sha256': '4b87cc23a414c9a98b5d96c1470941a9dcb60c3358a0860840d1169187589c8c'},
 'cylindrical_projection': {'ast_sha256': '7fd8b8c3a616d9fa51af6bcd17071c10d97f5905097c700d28cfab791070d19c',
                            'statements_sha256': '4275b80811594c587edfd50ccfdbd27e96941fcb88ae79890aeebece1516834a'},
 'finalization': {'ast_sha256': '4748b913d6dcc1f220be7fd407cb86f0cc6684a11c98af103bbe01df447add6d',
                  'statements_sha256': 'f91143b4b7eb8d448a8092488c9f62cc06aab6ed543dae888ce59e4dc6e6590b'},
 'geometry': {'ast_sha256': 'cc7d6c1265a632e04990fcff9a28b56aaea4d9ecc49a5f84924038179c54436a',
              'statements_sha256': '74ad545e755d1a9be58ce7398f4e88440644797502a75a54fc19ca241064c538'},
 'geometry_quality': {'ast_sha256': '323fa3284b32c45b638e7ca115f2d353a5e7c01d31f1b1d06e9eef6325d7575f',
                      'statements_sha256': '26e55569a85a3cbec880545762273d834f8be1ea96b78cbc5de10366a3618b9c'},
 'inference': {'ast_sha256': '761aad88a7bb5213a2a9b0933bdfb9cb2a74ff868445223dbaa8905757530517',
               'statements_sha256': 'b09a76c8c181f2f460716fe26191bbff31c7f41a5f85541bb8dbfc5063b4d9ca'},
 'interpolation': {'ast_sha256': '8eeee449aa93d8fb336329875fda4466cbbd8d3d35b08fc790568f0de0906a50',
                   'statements_sha256': '8480559611f49f2991dc6e91afefe76b09e27a3e1baf502d13490ae41c683a47'},
 'outputs': {'ast_sha256': 'f3970fee1e63ad227eea9891be321e77001b76e4adbb7d06a6e66107c82e0f0d',
             'statements_sha256': '0a5f4768153ed66c4525b543153fa6ec490a2f36dd8c4b8580d3d2e9530e422b'},
 'packed_publication': {'ast_sha256': '9b53ded3ae9b6d06c076a41afb97cce746fe2a241f4b7b9330e6b6254ffa78ec',
                        'statements_sha256': 'b872c097c8c33b587ca1459d0c4e9062604bb974d626cf62369955049c5e073e'},
 'pipeline': {'ast_sha256': '2b1105a5955fbe455742036d95c81d1e530ed64f52170c59f36a0461bda2426f',
              'statements_sha256': 'e4004808809d3f350716e555865acfdd35ec8cdff78bbfc72d87df1272c43661'},
 'publication_memory': {'ast_sha256': '411fb45ad28a9d0f9ab047d39a827ff76b83b4ca1f763ce5dcb02c8ec6392884',
                        'statements_sha256': '5dda8872d8df4122423a08397fa39fab99deebccb2757077c0cf824ee4b9a1f9'},
 'runtime': {'ast_sha256': '99881ee0f79d307a9d78a99aa4ee23023efc67de1faad1bb0e27c43144c5ed83',
             'statements_sha256': '740d0174f5cc63905efafddeb7302bd213cf525f37b417d3e356b400b834a83e'},
 'sparse_projection': {'ast_sha256': '293eea2cfbd358c4cd3e4abaa9b99011cd0dfb5722ebeb5f461a104b1132873a',
                       'statements_sha256': 'b51bb9aa3e648b6a830aa5fc6971ada9fa672c5faaf7e32c0b952a4238fd8285'},
 'spherical_projection': {'ast_sha256': '4f495f4da5d8d3083406e06f62a4cfb5ecab87aa9472cfb2f9b7f8a0b8878578',
                          'statements_sha256': '511900b81b66a0eaedb717f780b5c7dcabef20291d1d191576ebefefd7ecde9b'},
 'spherical_projection_cpu': {'ast_sha256': '816ff2baad8c9042d75d4a6048204d84bd982cafba58346d7d676e9dab6b53af',
                              'statements_sha256': '45a49858efabcc2cf09bb19611c46499f1bc8c6155a6e219fc3b629526538a77'},
 'spherical_projection_cuda': {'ast_sha256': '6ebb3b9c6dae32bada80c4aed3c4a2987dd81387f7d4ef698f2a47f7fac578fd',
                               'statements_sha256': '24ff7d8678ea9ffa75a648ff0f908273e04d08e4a860c817ef5431198df71df1'},
 'topology': {'ast_sha256': 'e78349c8f3def16f6ef44c66eff7cc61101a259203a6b5db7046e80a140af091',
              'statements_sha256': 'f9fc3fe8107900dfae6cfceed07125c23da4e5daee5939c11b19e2223b39c81e'},
 'topology_runs': {'ast_sha256': '2f7d46e6ff4b04c272de76e05787ec20347395a7fc4e85ebb91740cb6757512a',
                   'statements_sha256': 'e5155b5f6cebb7c722f6b6ffc1cdcd301eee990472cbe0cdc24a45524bb5e3ed'},
 'tta_scheduler': {'ast_sha256': '5080bfaf843d2aa3b900cfd49a9e904403f9607e51c1e164b30fba69de8e4ffa',
                   'statements_sha256': '90eb153729e9de672cec3525f76867561c2f839e4651085bc09866e6fbefcd25'},
 'view_prepare': {'ast_sha256': '11d2f0b0cb8534dd480503f6c58f30d0da3770cdd51ee22251115c57276ef7ca',
                  'statements_sha256': '8aabda96ce90c7f3ce4fb9392e223d81302a4ac3d80afd5e9e76b83376600abc'},
 'workers': {'ast_sha256': 'f82986774d22b50cef4d7b1effcfa0b8c806f56672b171f0c946ae8327517922',
             'statements_sha256': '695d4126aaab9c770c216141d2a423a2dbab2c9a9fa5fef71a94c1664af08b0b'},
 'workspace': {'ast_sha256': 'b11e4b5d91b7fa3a009db6536e5a70e25cb5455ee67e59a22076cd2868ab0196',
               'statements_sha256': 'ff8415543d6ce584190138ea0badc483c1301268216f07bc382fabe4c2d22c7b'}}
REVIEWED_V24_0_4_RELEASE_REMOVALS = {'definitions': (('assembly',
                  '_SparseComponentKernelUnavailable',
                  34,
                  'e111e9defedb96cc19e0b2a9cd9f25f682a031d955bd9399edfd3dc53e34499d'),
                 ('assembly',
                  '_run_sparse_component_kernel',
                  35,
                  '19d5deeabff143086177fdbc7647220589ee4b002077930ec8f67b1d887a11c5'),
                 ('backprojection',
                  'main_process_gpu_stage_inference_overlap_enabled',
                  40,
                  '9fc3219820cf9ac003eecdd9c28fafe224c7ec35241f6c7928d1ccf73b946648'),
                 ('backprojection',
                  'main_process_gpu_stage_inference_priority_enabled',
                  41,
                  'a2518d42f5309f20de5f63a51b447f64b0f9ef4d899bf6d92299db60feb2cb38'),
                 ('cuda_backend',
                  'gpu_cube_resize_enabled',
                  32,
                  '85f46a419041e64d6818c25686a69debb1ce42eab00f1f1a3537136e4dfe590a'),
                 ('cuda_d1',
                  'raw_bbox_nrrd_layers_enabled',
                  92,
                  'af5318ffedcf9a7cd04a720b8373f480308d43b379db96aecdae3b1b195c6dab'),
                 ('cylindrical_owner',
                  '_bucket_shell_pixels_numpy',
                  27,
                  'f247ed177d854a52dad612649a85d5ee3f19d81f427e8eb19b2acb9edb586f68'),
                 ('cylindrical_projection',
                  '_occurrence_rows',
                  52,
                  'a30e06657361bc2dfd8686b405ca0810177a9565bf6628833c3e95226f964ea8'),
                 ('cylindrical_projection',
                  '_pull_radial_chunk',
                  53,
                  '8a83c029f4e00029ac4e46a939aaeb414152751c474c14b6c009f0b9bef46892'),
                 ('finalization',
                  'fused_final_native_sparse_cpu_enabled',
                  37,
                  '1f75304eea8609f17205b3ace1c1c3245a5e4a8ca696230a4768ad24a6432af3'),
                 ('finalization',
                  'fused_final_restore_geometry_groups_enabled',
                  33,
                  '1261fab2d8c1e0300731b8e25acbee77e374c17c7b1f8be1dabe7894ce978bed'),
                 ('finalization',
                  'fused_final_view_union_enabled',
                  32,
                  '98dd60e362abd44a56e912ee5f31da3cb0a24bb79ab468cbf3cd28439fb57145'),
                 ('finalization',
                  'scheduler_push_drain_enabled',
                  30,
                  '327e1e86508b9bb4f4ca3163ea257b7a8fc2eb55e5828467c1638298c32683c4'),
                 ('geometry_quality',
                  'spherical_cpu_compiled_requested',
                  9,
                  'ed49d8dd0ff2a2fb68abaa0be6013938e1e475f0499aa3171b065b1578d23c45'),
                 ('inference',
                  'cpu_retina_roi_only_enabled',
                  64,
                  '4e614daa55c1bae2b42cba583b4ab987693eb3f63aaa91d3f1ca1fa894e9a61c'),
                 ('inference',
                  'gpu_retina_flatten_enabled',
                  27,
                  '0213ea156b70211d6a87ce12a391c0ca7ebf39aafa2b1fdffadfb6caa71965a2'),
                 ('inference',
                  'gpu_retina_proto_union_enabled',
                  32,
                  'f93aad94733968277b44cbde9834ecfd6a386273c3d2d059e765450e98376839'),
                 ('inference',
                  'gpu_retina_warp_enabled',
                  28,
                  '7942c5b620aae4ad140394d84de934a0c16dc018abcdd903c067c940719ac8eb'),
                 ('inference',
                  'gpu_worker_chunk_hole_fill_enabled',
                  105,
                  '681958a8a28246c1be2787bbc1df67b5c227cf48203c59c7e1ce3792e9259a6c'),
                 ('interpolation',
                  '_disable_planning_kernels',
                  53,
                  'a35ffa4c28d8f648336453bbfe85d21ba01d3e83554838cb3e14184bf3c731e8'),
                 ('interpolation',
                  '_find_slice_projection_candidates_python',
                  59,
                  '2964a4a06f74b43fbdca643ab9663fd3332fa8f854a2ae69f3fae162a7775dc0'),
                 ('interpolation',
                  '_planning_kernels_active',
                  52,
                  '7491461cd2145b63c672cd7ee691553615fa1a59a8055304c96d1c265f0f78ba'),
                 ('interpolation',
                  'compiled_interpolation_kernels_enabled',
                  50,
                  '30c0f53d9d39495eb1438937cbd78e328c70101fd45fa49959cda7f3969a16b0'),
                 ('interpolation',
                  'compiled_topology_kernels_enabled',
                  49,
                  '249f4dd61db7f15e380d00554d7a4d46704ea7fa3acc85300f4519649af159a7'),
                 ('interpolation',
                  'interpolation_fused_bridge_merge_enabled',
                  63,
                  '1f3e7f920b8f332e09fa274a4ede2f928206d787d0ca974c0c5a622def710fe9'),
                 ('outputs',
                  'nrrd_extent_zero_skip_enabled',
                  190,
                  '6faddf850a6ad2ef9b06ee2a8a278c6b6df64f914bf0dafbb991f3078c664e17'),
                 ('outputs',
                  'nrrd_live_global_layer_enabled',
                  174,
                  '4129c3a04054c438da6ad1a90ae1ef3bdb7370bddbfbb919c05c0f2a0b16ae76'),
                 ('runtime',
                  'gpu_worker_direct_union_enabled',
                  88,
                  '076e77d722c0fdd7511d09611ca93a4b83ee0fd02bd4992f45779fc8dffd21a7'),
                 ('runtime',
                  'hybrid_gpu_stealback_enabled',
                  107,
                  'ef59a415eec65371504f7c4539382839cf9f56149f8501c4dec974822214cb5b'),
                 ('spherical_projection',
                  '_nearest_global_shell',
                  36,
                  '1991691dadd4deed0d14403caa2f784348b0fa6a0376c8a564d803e864be9461'),
                 ('spherical_projection',
                  '_processing_index',
                  37,
                  '81a698cea7313e8069b2e4238c485379dcaf365490a402b12f3f22a169a70191'),
                 ('spherical_projection',
                  '_pull_spherical_chunk',
                  38,
                  '95a0a2ddfb7bfb44b73df1e9da283e7ae844d59a6c2c008af32ca4d1f195c38a'),
                 ('spherical_projection',
                  'spherical_cpu_compact_enabled',
                  29,
                  '4754d1b5eaa22c75e2e6225c781806dc8666172e3f71c240006f2dd25f4a607c'),
                 ('topology',
                  'interpolation_skip_compact_relabel_enabled',
                  41,
                  '599c922aa6dea6b8d3951542a0a7b750b05c40e7864bac4e377443b7d949beee'),
                 ('topology',
                  'interpolation_sparse_labels_enabled',
                  39,
                  'a24964b4546a08355976a241f5c317fd7e24a8598777fce0395afae0d0e03869'),
                 ('workspace',
                  'tilted_inplane_linear_enabled',
                  26,
                  '974e196d9fbf2d0128b3ebc8728c9992a4c02a54f21a5d1f880d7877ec8f3993')),
 'statements': (('_deps', 2, '14fb4ac0c90c87c6b4c672e2bf3de36cf41f934438645a7cc37129c0e1bc80f8'),
                ('assembly', 32, '8b8e9eb230fe2b63c9496772d6bb7ad340050d700bd52aa82f5ebe5fb999d5e7'),
                ('assembly', 33, '14726e24984d9df4f8db59388a57016387e1e658052a8f551afa8af37648ce67'),
                ('backprojection', 105, '138fa3acc84bf3ba73c42ed3853ad547e780d51cf25c6107fef73e51a9266351'),
                ('cuda_backend', 24, '2de8f9f12f3d1012fbdc7e18f12ddc8b636d37e529b8a10de20d6367202565d4'),
                ('cylindrical_projection',
                 42,
                 'a00fd59c3a0ef57a76f093e34ec03927219de47f7f0ad44fe7c22b21bc713347'),
                ('interpolation', 24, 'c073e3df6f536331f044435d5b85b818d00c09ddaf5e7c78497d7f43fea6ec6b'),
                ('interpolation', 47, 'd24f27cb632054fb67662096b8c4070083fc9c3b985370cccd7d26337306a9a9'),
                ('interpolation', 48, '3ab30c336abd35d22525a428c7d84454e0105acfd26b72e24434d8616cc10208'),
                ('interpolation', 51, '4e01060dae0bb88826bf2b113bcb0f5e9d10f2ade407861176dd712774bb430e'),
                ('interpolation', 57, '42b6144e5eaac69f1d042276d509eb9d9e470d16f848cee29f0c01a51a8b2253'),
                ('outputs', 185, 'e7dff3fea3e8f03b78266fb443ee7ff6a55559fcefce4003e4ea192bfe2bf6d9'),
                ('outputs', 186, '29265bc1808725d5378e1c5865e0bc9e926d57d3ba33f0cee556ee002570c45e'),
                ('outputs', 187, 'e2c64bae2e96424a93052b292875c466a1a6c500d60a1e7bd51c4342098f31dc'),
                ('packed_publication', 6, 'db93f1d46cb61813da7a6de5c765048e75a2feafc0c52bb16feff796e25e999c'),
                ('sparse_projection', 28, '1d8caaec63bd240e132cc12a2f9a591faa7f6974047c46ef91d7d76a91e3dc33'),
                ('spherical_projection',
                 13,
                 'c886e8eacee2cc9f88f09bc14036d0a53628632deda64d9a24c334cdcbb355e5'),
                ('spherical_projection',
                 14,
                 '8cae185e0005fc2a9d67faecdcc4873d23efe59b5981e58895b612fdd853ae4e'),
                ('spherical_projection_cpu',
                 8,
                 '62dfc204b213a46d14451a67a59b2d9ac8d1625a929df9f22a601b16be349147'),
                ('spherical_projection_cpu',
                 9,
                 'c2ac2fee3ddb3649c691ab29da24c51cca922f57e65586eeacb81b759994965b'),
                ('spherical_projection_cpu',
                 10,
                 '63020afc675eee8a36e437c13a59a4b57c6a23a77655ef87114867bd4a20ada3'),
                ('topology', 21, 'e7fffed290409ac21b8ea3f3619a087ce96d51576973e806684dbcda5e43ca8b'),
                ('topology', 22, '34e3971f0db977a8caac8cdee85f257d108adb8fe05c301f0f5ac657a0ce2238'),
                ('topology', 29, '550c6f6ee03b1d2ed4339c862cd934c6650182ba4eb44a2d3a27792df064d4f3'),
                ('topology', 30, 'e788bfa27be8f551f0df192472371bcb65e128ed9921e5f6f11e93655941e274'),
                ('topology', 37, 'eaffedf3ccea2cc63cf7173a998341012d7c8f78b44f57a3c871a1e7ad3edd2e'),
                ('topology_runs', 5, '6824f9609888b53198dac49de1d9ce860e0894d8c36a2d16c926c7a75a872ddc'))}
REVIEWED_V24_0_3_RELEASE_PREDECESSOR_MODULES = {'__init__': {'ast_sha256': '54919b484395b8841c680ff2be9f22bfde9d3534c9c2a345baeb093e6d066ebd',
              'statements_sha256': 'b553aac5798ed9f20ae446abc38c9c7fd172f0cfcef74c1a545435aa4722929c'},
 'cuda_d1': {'ast_sha256': '2ca87f7f12a5524f1ddbaebe9e5ff92e2928199461823fb94d3edf64e19b6ecb',
             'statements_sha256': 'a3e7a0be18621098c8496fd9733c2b835bd3529def8cc8f8b3376419c97686fd'},
 'cylindrical_bitset_compaction': {'ast_sha256': None,
                                   'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'cylindrical_owner': {'ast_sha256': '39538c0eb4c3c06078739d0389d17646900fa8c7f38ee94436120bc372d6cab9',
                       'statements_sha256': '7ccf3338776127d44c0abc3751dd23d3c4c2019152eb05f8c082d84c1f491709'},
 'geometry_quality': {'ast_sha256': 'a650969a11a2d4ff1055f4f5d2c525f890227376dcae1bbd73122209e53f0c86',
                      'statements_sha256': '352dde80efab4ac2fa9aa637a7bf410efb3f1bf129657ecf78579931862fbfcd'},
 'outputs': {'ast_sha256': '62f54c22619f869756a9813f32570fb9b7e417514f6e7dcd48c539d230a15df7',
             'statements_sha256': '966f010bb0751eca559203610036fa29263b961e01ce56bb6d2b04ac140ab3aa'},
 'spherical_projection': {'ast_sha256': 'fd6c525aa953209226a85d48faa9e840c9144fb2eb7e00e71ff0949f4710ed5c',
                          'statements_sha256': '764d261f31cbb11793554640c9cf5f3babf5497f3e20588311cb9add404c3fd8'},
 'tta_scheduler': {'ast_sha256': '8328d4b7853a53711886bb961a606729910049fc2e44bd87646c2c5c9120ed8c',
                   'statements_sha256': '2f384295a0f8115012a184c1582fe802bc429801ad3cd73744c60d54cd7c2383'},
 'workers': {'ast_sha256': '0986c8832340ec5f31bf1df1e5c07659e6757b312bb683405c41341e695daecd',
             'statements_sha256': '48acb4ac4db2c06a6e0022dd5dbd25423ab27e12c997a6d27c08adbcff560587'}}
REVIEWED_V24_0_2_RELEASE_PREDECESSOR_MODULES = {'__init__': {'ast_sha256': 'c551075f390cc95b55b867b073b7329facb9a550d308a2911ffbf8ccc22670c4',
              'statements_sha256': '980addef21956e2d9f06912c2058f2b8349dbc8c4800d92eb2e3e8a9e54f7871'},
 'assembly': {'ast_sha256': '829a5fcb92fc05ade5b5a33d737f570a076449532de67215ec3c7233f54883e8',
              'statements_sha256': '594003943191ddf570a52c69a03e3c048a0be5736f9677adab2989eb1f65ac86'},
 'backprojection': {'ast_sha256': '36c351f239e1faa04322a1839ae8b1259897002d3ed146c0f702e73d61641a45',
                    'statements_sha256': 'a9eaa5747f2a347e5381f88d2cb285969d48762d5d1ebb41ddb3df45707d823a'},
 'cli': {'ast_sha256': 'c8bc79d00e12e2a447a301bdec61aa01361818c0489ec524facef093d5d6bce7',
         'statements_sha256': 'e4331bdb2a81d4a87feacfcdaf4cc8f420b6d90fedba4f1256e5085f858cb537'},
 'confidence_evidence': {'ast_sha256': '46b4d73dfcdbce2d95f199626e95eb004fad972dd0d79e9321c2e8ba13aae4f1',
                         'statements_sha256': '80bba8f732ade54d726453b86a15e44a3ba51fb91389c4a65cb7973219f3934d'},
 'confidence_native': {'ast_sha256': 'cf1e1b4f8484e817a06e52a0d05644645772d5244d4740084352d5623095f760',
                       'statements_sha256': '19b6e8999aceebfe3e91ce02aed0c4f58b1d7dc6dc80798e9526d187f05cef81'},
 'confidence_projection': {'ast_sha256': 'f4a912577c964ef0a1b3714e14b6e6c4eb4c70cb8eee1355e99f4286a8b2b3c8',
                           'statements_sha256': 'b3c8420b6c0136d60332c25ff2c83904fb2fadfe2261d2b627c305e30756c03e'},
 'confidence_tiles': {'ast_sha256': 'ccc63ebc7632d8d15eb0f19c5b7eda0ac4a58fa0e55229219551e5944b5fed5f',
                      'statements_sha256': '1018a4b3130f5001d04f5a14b468a441c14cbc37d099d32676a38becabbc806e'},
 'config': {'ast_sha256': '1d425017bb6d56cd2e09e017f8c161226f9da981b5b1e655a6515261e32e9651',
            'statements_sha256': 'd3b06b2178584a9e399aa7861115b4e624349b01bca4ef55cce34a2da092150c'},
 'cuda_d1': {'ast_sha256': 'fa2a3c38e16cb5d27f15379cc47206f3a33c939380e7ccb957645ac4529c1910',
             'statements_sha256': '49b6a9565e8a2456088743ab088582d1b00ef42f3bee174c271d50d5c10b91a3'},
 'cuda_finalization': {'ast_sha256': '48e8d77afa99044aff5c62f0f2735c2a004bf13652d67fd4fbaa91e4ea5e80ca',
                       'statements_sha256': '2e2e3c8f13af942a5ec2c8c11b9d15eacb8bc9cb89f4c9257f9ef5c793bd56fe'},
 'finalization': {'ast_sha256': '36af03289661311d7cc280e6d537882cc961be9c7ed2534a85f6e85a50891201',
                  'statements_sha256': 'a2102bf5c34c3a9b7b6515849b09ad35bdaa2f391fa3fad7892ce5f4dfc4b756'},
 'geometry': {'ast_sha256': '0e559cc1ee9ee975bffffdaa7fb70f4bb38119ad7640b4defde0ff7aa0ed71bb',
              'statements_sha256': '7f2240fa84b9bce917401452e4c4814fe809e188431e43dbf560e488f21a9317'},
 'inference': {'ast_sha256': '646af699c7314a4b0fff1e5568a86313d7074acc5ce9f71a2307c70fe408cb0d',
               'statements_sha256': '9ea5c68a48769e7e4dcfbb639982178f545b5ce454bfc51cf066d0842aaf61b2'},
 'interpolation': {'ast_sha256': 'feb1b8d59527520d5f3cee1f0b4244faf3e86132ed0f424d0903658f4fdbd723',
                   'statements_sha256': 'b243a97a95bd7887c600a8928df2bcea68b4059f327f26aeb3c9a3da86c42175'},
 'json_publication': {'ast_sha256': None,
                      'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'lta_execution': {'ast_sha256': 'a53756a94f20212dd5221ed62addcb4f31c415e3976f161fd466082854275385',
                   'statements_sha256': 'f216bdca5f04ef531c27d5264f01d1094c615b1026a6869d17f8f9374cdddac5'},
 'lta_inputs': {'ast_sha256': 'aa84f97b5266c05867a92308f902e3d7ece490366fdd6de12ae5401fcf9f3bb9',
                'statements_sha256': 'c9991ac94be59bef53a54742b60c3a3520e8d3f63e0d27b298a649d5e0d85d91'},
 'lta_outputs': {'ast_sha256': '11f84203e5ae173002593a4bfd422550c439c6c14970e3e2dbe3bb70c59ec3b3',
                 'statements_sha256': '28f34cab0df8dbcb1f2dd1976e4f1a2f877b6324081afbc6e703b9d61792532a'},
 'lta_rendering': {'ast_sha256': '17c74d08b6b4dff5f30653bc6fbe904d471cabd93232d969dd79cc7cbc632717',
                   'statements_sha256': '88e7cb3c5be663c09aef9bade839759cc943c034cece394eb5032514c19e4f2d'},
 'lta_sam': {'ast_sha256': '3b904d653ec1423440a976199350631d0284582e4df5f0dcb4107eda248969e1',
             'statements_sha256': 'd38aa13eee71d52359a5bbf13aa7a8e2135b15497871825961613ed5b32400f1'},
 'lta_union_artifacts': {'ast_sha256': 'cfa1d3cd064f75fab7e318b744a3bbfb864036b17946a7700cc5c8b1c4ca531f',
                         'statements_sha256': 'dd31e4b5d225dfeb8808827274d21447fd1269137b74f50588d50bca189167c6'},
 'lta_worker_adapter': {'ast_sha256': '6dca0a0b6257802fb5eb7267019022480ff57a0a616a57e6808f56f829833777',
                        'statements_sha256': '60badff0a69afba31ea40b366539153376ee9f6e1416170de974c2ba86991e75'},
 'media': {'ast_sha256': '7c14dd99bcbc3ee02b0d816637ac3c51d76d12030519351510da23a92d213459',
           'statements_sha256': 'f7d7d9eb1dfd2c3c6e7e0e1244853af50b86cf8687d22ca29996e2bea2cd4205'},
 'outputs': {'ast_sha256': '130830f7b0f4c308d32eb2d45954e76be156b4e3b1ef303ad40322f907350137',
             'statements_sha256': 'f30db29543210192c9c7afa131f977e7036bfd4396c42fb49bf4edf901d00118'},
 'pipeline': {'ast_sha256': 'b070a8a8d45f52eb565d898e765c87a208fa67c4cf073e251ae088314901b828',
              'statements_sha256': '5704f6ecdd5bc2ddb5c5163fffa6cd07a17457b83683e0e8d5d955a7e34a624f'},
 'pta': {'ast_sha256': '412afba5daee14d90f56b313f22b2e1a40ac6df1f21cabdc3614838f5a9e7680',
         'statements_sha256': 'dd76e5fc9b74097c53ed6472989e40e4e7b5158d75b3c2ae32f44cd202f9f55c'},
 'pta_augmentation': {'ast_sha256': '48e9af8c4298dabe827e5c65725afd609b258fc8235d0eb827f3f91dbfa1d6f9',
                      'statements_sha256': 'a14d633fbf5dc03a5b23329ab3abfdb7bc64cac9de323f5a877735d359f3548b'},
 'pta_workers': {'ast_sha256': '1e7b3a59f094592ac1df7f0e7f09fd184b220dca099f67d32cdfa326ff7eba1b',
                 'statements_sha256': '9e3b107800ce42ada5ad42984f2d0d3752706559082641550109934bf6ba88a3'},
 'reconciliation_runtime': {'ast_sha256': '67ce296a9a3455026c3550ad98db2bc9f9da4ce38cbc3a516b9555bc8a82c8d0',
                            'statements_sha256': '9f68cbd9957c9bb69e62d8954db361bfe3719e09255d93a4641f15670d44b65b'},
 'runtime': {'ast_sha256': '1ca01142cf1eb9fc226bf6f0e408cbdee5ba0c1b2048e940bf864ce7dd320516',
             'statements_sha256': 'be5455657bda035b0158aa1c057b02dbb0d647f4b28100acc6af183b1f925646'},
 'sparse_projection': {'ast_sha256': '0168bcee8d95f9f5e7aabe152b5fc6ab9f56e16f88402e1190282d0e0ad4a792',
                       'statements_sha256': '02a75eac336a59aa56f9911f9202dad201cd407fad687d5e397fed60a08f5165'},
 'tta_augmentation_cpu_runtime': {'ast_sha256': '91560a6ea90e09f95bc779eac137c19333a4e26c306f3a46ddd3eb416ea04dec',
                                  'statements_sha256': 'e7db1041c179e8bcd834f465ab59797be80e2b7ec15daa9409c6ae7c311ab107'},
 'tta_augmentation_runtime': {'ast_sha256': 'ecc66621d2a405dea13ed0f3e66d50280f67ecc3646519ce8f08bb9b81d10c66',
                              'statements_sha256': '7e47ee5226d12a92ab84fbe549e6a03a5c4ace536a5b642031698ff4eaa2c5c1'},
 'tta_outputs': {'ast_sha256': 'd28baeddb14a6b32deeff194123a5d9222e910e1b1090d3d4a996e3b4ac97204',
                 'statements_sha256': '59d17afe9a02bb6459ecdf2345058747492c8547bed97fc9db9e822f83c66352'},
 'tta_scheduler': {'ast_sha256': '7ba34bdcb3ba728e21f17968f2814a268d489cc9c61ccf1d1c3f33aed49e9367',
                   'statements_sha256': 'b2358d45034f2514e38d426327adf1d59b9e01f23cb5ab86d36ebe88eeac2664'},
 'unification/manifest': {'ast_sha256': 'e640ef35b970ddd8cd62307762070857c9c376c0da375435e19416418eaad606',
                          'statements_sha256': '45fe05d97e7911cb762077488af0fdcd80b9807aeb35de634e4d1ddb7418cf37'},
 'view_prepare': {'ast_sha256': None,
                  'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'workers': {'ast_sha256': '4b0a8f17f1b7820dc04c3c4ffacbff6d75d81ba0abd9543facfb0e48c0466ac3',
             'statements_sha256': 'afab2f850fc5b0b71a5cc1f408660bcd422790ff500d58223b5d1612b3760540'}}
REVIEWED_V24_RELEASE_PREDECESSOR_MODULES = {'__init__': {'ast_sha256': 'dcde1466868161785feb9d9a4cdd945d657e659acf3254fd1c098acdea90f855',
              'statements_sha256': '78f5dcfdf216ef6cae28880ce30eac9e5cb9895f9ad66a5318c341649c758e13'},
 'cli': {'ast_sha256': '78e224a912de718f7d5e43647ac20823d15594e54924acc89efe08208179b3f5',
         'statements_sha256': '711e7f4a4d0d69d67d9331c87fc5ed64fe8c084e3e0b3276f2ee2ae26835380b'},
 'confidence_evidence': {'ast_sha256': '7e24732607efb4701b178021c6ae70c24ffdc08bcaabba12ba62e45f8a99405f',
                         'statements_sha256': '338e243e872e5d8998cf0bf7a45e817671b70fd98f43ce7801e049ab965a6b86'},
 'confidence_storage': {'ast_sha256': '20fd1041f16e2857837792c088ecddc9cffaec67c5283991c8f5631a253e995c',
                        'statements_sha256': 'c053637432086dc017d2c735326eba92ab4eafd3efe9232b77576d7f309bd967'},
 'config': {'ast_sha256': 'db46529eee5ff196f946d535d7c72ae98e2331d44f37450c24b0615741b81fff',
            'statements_sha256': 'a9708685647b3797694694e04849451a3a984308e0e01cd07278499121f86355'},
 'examples/external_augmentations/GPU_baseline': {'ast_sha256': '5d4cae99b413cb630c709bde278b552053de818c81719f258fd11d33ad5d0b12',
                                                  'statements_sha256': 'd06c35567868fad2399b8e226f747ab850f47c308fcf7b71eef2c149d39dd5b3'},
 'examples/external_augmentations/GPU_heavy': {'ast_sha256': '0ef1a96dcc4f8f73a88e4927e587a03f89089369868b1fa18a574e8144335f91',
                                               'statements_sha256': '862a5d8f3f0e130003d792f39244989589716cae4e3a04ddabb73bfafcae31d1'},
 'examples/external_augmentations/GPU_light': {'ast_sha256': 'b39fe374ab7bb6dcdf5a1f712ea5f7a0e12a27b1339fe081b8a1bd4eabd2b8e2',
                                               'statements_sha256': '04a9ac28cd2178b289467d70bb8a973ec4992af4838fa42caa4a2be4b5929b1e'},
 'examples/external_augmentations/GPU_superheavy': {'ast_sha256': '93b3865c202b264232a39e4edd7152ff28413446f447dbc873ffa0324af4f915',
                                                    'statements_sha256': 'c6ae28978100512baeb8a4962a1cabd9e5f73b16fc885f3cdecc4873ab42f42a'},
 'inference': {'ast_sha256': '5adbc1f51297261a2a7e379f02b6729cfd6424cc3e00b74750d1f0282a11a367',
               'statements_sha256': '826f7c6677f4af8d52aa7550c5b8a0d87cc312523c16a5989726e1609e963a2a'},
 'outputs': {'ast_sha256': 'daef7636f7b692a5fe8c28e72b217c9d6686fb76e7095a631b41ab6c2c687192',
             'statements_sha256': '2e852f162dbfdaa6e75e136e3a5807d58377b9136afef5140848c5f87a8d0e1c'},
 'pipeline': {'ast_sha256': '4ea26a9c49eaf45e238a5ee160dda37a2b5ad80056075012129cd41d5e8fe460',
              'statements_sha256': 'bb527d27902be43edf6e80068329bc79796b71cc2dfb9b85dcb5f9688af5cb62'},
 'pta': {'ast_sha256': '7128c9b41e03bef5c1591622a9c6730d66137ebeeff32bc62f3da1387e3625f2',
         'statements_sha256': '406aa002954cbc2240f0ea2aff8d75ce03ed8b12c6a457aa927cf5646d42ef43'},
 'pta_augmentation': {'ast_sha256': '03c9a6df7a9e268ce8c228128a79819656a8c64fb8cd19a6b7ca0d624e540b08',
                      'statements_sha256': '1bf4603195cb79ab5625a23227c38d2534a8eb53c9a43733367bbb806480280f'},
 'pta_classification': {'ast_sha256': None,
                        'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'pta_config': {'ast_sha256': '9f4522ed5dcc3691ad36e007a9e55baf25cb43d9402637674ba9c0031f6a17e2',
                'statements_sha256': '7c82d9bed00005b763abebd95fa37f6d2b88bc440a144de951cd6ea7e9a58565'},
 'pta_cuda_azimuthal': {'ast_sha256': None,
                        'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'pta_cuda_cartesian': {'ast_sha256': None,
                        'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'pta_cuda_masks': {'ast_sha256': None,
                    'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'pta_cuda_shells': {'ast_sha256': None,
                     'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'pta_gpu_publication': {'ast_sha256': 'af1fed979738c1dbb597141d252236abfd5dd9da5311a1ed00ef0894a0cec7d5',
                         'statements_sha256': '12236f43925ba2194e0abaf86ba7063ecf82683f5975e92891db1a5f8d2a2c80'},
 'pta_publication': {'ast_sha256': '6fa72e51b6b00d3a0da057ea21d9cf55cc2c8280f1d5e5455b88c1ab5a7c38d9',
                     'statements_sha256': 'f815844f92ada52c2ab6580f506a3d0395c32aaface66327fed8ceb15e7d7b05'},
 'pta_runtime': {'ast_sha256': 'ce43a37e08f314e52e479ff33d5cfe1c6bc478327a1b0f716c1b0399f0cdc05c',
                 'statements_sha256': '34c42ac17c3a675c37aa7f77fd8336df6b39f659dbee50b1b1d7155a7cd51090'},
 'pta_workers': {'ast_sha256': 'd1c78e69106b7da76cddbeb38a4d00e4f05ea41f786d4f27054c67cf32b3c425',
                 'statements_sha256': '995f15b0e858088044520351de177d88a24482f7bfcb09cdc3557e80fd010a71'},
 'semantic_cuda': {'ast_sha256': None,
                   'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'semantic_inference': {'ast_sha256': None,
                        'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'semantic_trt': {'ast_sha256': None,
                  'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'unification/sampling': {'ast_sha256': '1399db3603cd654c50306997af8d3d30606a5ccd2a5c78238aa6bdce96d3e64c',
                          'statements_sha256': '341899d198f844623df60b754cfa6e4a2c2335e4fd924c8247c1d060c5aa17ac'},
 'workers': {'ast_sha256': '14f3eb325e4b08817b5b7427f9b34098a943ee394d3284fb93967474d6ba2a06',
             'statements_sha256': 'bd913fcf90cb0253d4283e5825f1104504a11a58ee437c99a184aef30129d4d5'}}
REVIEWED_V22_3_2_RELEASE_PREDECESSOR_COMMIT = 'a70e424d62916d2e3ee0ce2dcb3d02565a6ccb3e'
REVIEWED_V22_3_2_RELEASE_PREDECESSOR_SHA256 = 'a89f89a577b43fdd9f776a1aceea7f22354c7c7f29d2e7c934319d23ecb4a6fe'
REVIEWED_V22_3_2_RELEASE_SHA256 = 'dcf330049b7d26f219cf96b723d6b4c8368cb9eacad4ca715b867ee1528864a5'
REVIEWED_V22_3_2_RELEASE_PREDECESSOR_MODULES = {'__init__': {'ast_sha256': '3c3539c6aa80ac164264885b0744582c8dfee86d5fc6660f219f654760bebb12',
              'statements_sha256': 'd159a370a1aee9b6021d41a5f7cdb0605441f80e38f53ee7f21d2bf08230dd62'},
 'assembly': {'ast_sha256': '42fd47d7dea698e9111b63cd3a163932804651526a5f464a6badb7b245f6132f',
              'statements_sha256': 'ec0065fd3862bdacc260c1bc5b0e69ff9948a50bbfc82ea2bd648798b819d84a'},
 'backprojection': {'ast_sha256': '4f5d45b9144e610c88fcffcb19119c284b7b8adc04a24b69cbd86f9d7aa5d390',
                    'statements_sha256': '78f5bb572bfe883b8bf19adfd6a62e9e4509c45ddda109803b8f9b8b1773244e'},
 'cli': {'ast_sha256': 'a8d4b2f13da4000495c3846adbe50b9a05d7808e65216cad9ff4bab89710df5f',
         'statements_sha256': 'eab76d404307a25f27d492d74e50bde7bd32c2a3bb1076cd7a4d15115d54c064'},
 'confidence_consolidation': {'ast_sha256': None,
                              'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'confidence_evidence': {'ast_sha256': 'f6b7112fc162da350e2cce66c80e18a5b57446e6325e83dfa708615494ff7cb9',
                         'statements_sha256': '86802fd026e27e6fa8291931565b88d2ecb9a6092e9ffe4e5a02d651649898a9'},
 'confidence_native': {'ast_sha256': '19bd2ff313f4e83a69c6e156a4b4d3b78cc6dda3e72d0075ce1c90651d47d323',
                       'statements_sha256': 'f866b480c50a1fa367197770c7cb5913243f6acc6f82cef1e1c64d3c69fa5884'},
 'confidence_projection': {'ast_sha256': 'd165fe6d8c3e0d0e4adeb0492c68c3ddc2ea0fa385e6614974618b51213bf32b',
                           'statements_sha256': 'ea81467be6b485682a1d3648dbf12e8e725e97915715df81827dde0b1b929598'},
 'confidence_publication': {'ast_sha256': None,
                            'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'confidence_storage': {'ast_sha256': '3870062f0be6c2977994f3ca4129ddb02b065e4f991d16dbfa7c3287837f8ece',
                        'statements_sha256': '49a4d5bc16d75f296693cc08a571fe72b33b9f5001e5882a05fe1e6e0676ed07'},
 'config': {'ast_sha256': '920b53ccc53b68cec6203a4ceba478e1becb1bb22b56042dad6c10aa84875405',
            'statements_sha256': '75ebd0d6dc8c811d7e19a5127963af98ff73b332706b0d86d091e17d4204a0f9'},
 'cuda_d1': {'ast_sha256': '7164457a01b1fec9826651fbfba0a64a0e1f98c1d4a91ce37d0af0bd4f9e226d',
             'statements_sha256': 'ccec39585c707b5db612bf8b6fdd8dcca64b588bdbf90ce9759a890c225b293e'},
 'd1_confidence_retirement': {'ast_sha256': None,
                              'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'inference': {'ast_sha256': 'efd9d52318ca545fddb318da2723bb29955bd2b0fecdf60a875555bfb7e8e396',
               'statements_sha256': '122d9dc68f8aacd64282fbbbc97df64996ccc9d10d40d6927b99202e848fd87f'},
 'mmap_advice': {'ast_sha256': None,
                 'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'outputs': {'ast_sha256': '3ae44fa124cc10ff4c40133bb7cca6f6c8e33fb418fe7e07fbc61ebcb6ccdcc0',
             'statements_sha256': 'e4ebc30f4fc8a4f43f8971458d2f08d10b4aae9a565b166b47fd3fc78220c88e'},
 'pipeline': {'ast_sha256': '87a62706845202935b5d22ed47f86e95b98ce6e532d079f15d05e2a3e1d422db',
              'statements_sha256': 'e98303f6acd71c96042ca5381aa21ff289eb577053512c24ea46f35497bdbaa2'},
 'publication_memory': {'ast_sha256': 'c780db7c384ecc96e51bec7bc1e9022de029168604e302f0d24c148ed4885734',
                        'statements_sha256': '5451c79045b911ddb99391e421c929c2a2d35f67509394e5c84e5cd13f02e8d6'},
 'reconciliation_runtime': {'ast_sha256': 'b198c65b02dd1401c2858fc05223cd7d00a995f91baf281cc6e5688d732a403f',
                            'statements_sha256': '3407cd2f7b9adbe1c8b43a867854681fb6ae2d3dfef5bd03a6a68075b7c858a0'},
 'runtime': {'ast_sha256': '23d81bb56e5e4b3b394dc85a1d6cb17cce7e25032d0ea04990974701e734f591',
             'statements_sha256': 'ea407ad1247732e103ede557bc3b7b48732df95a8d4066160f373fb071d6994d'},
 'scheduler_diagnostics': {'ast_sha256': None,
                           'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'spherical_projection': {'ast_sha256': '8052681c83736eb501766fcab9a0943b680bc4355d0cbf2820d6afc6402a27a4',
                          'statements_sha256': 'f99d2afbc5141b1cd96375f0d8133f91e3d999d0363bfbf4e391fd52483ee497'},
 'spherical_projection_cpu': {'ast_sha256': '0dc03f373093400b48bb3d1e19391392eb5ac75c170d1046d592abe8bb332770',
                              'statements_sha256': '0eea28ec567d27910519a229cc23b4d57817a6426eedbd91196ab95130a5ddcc'},
 'tta_background': {'ast_sha256': None,
                    'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'tta_scheduler': {'ast_sha256': '2e5b0163067b35a0a3c7db53f456476663cbeb22a19d44c57a01d1c15af16bc7',
                   'statements_sha256': '5ab09f6c15b8d2123f17e9fd22a2831f67ff837b1c91dd3e38ec776ec30f0c33'}}
REVIEWED_V22_3_1_RELEASE_PREDECESSOR_COMMIT = '300cda53b477bdc2263a80d53bb354dc2ea8df46'
REVIEWED_V22_3_1_RELEASE_PREDECESSOR_SHA256 = 'bf3febad0753470694fa64f88990b72f24c7b45449ae41dca2beea43f49d82d0'
REVIEWED_V22_3_1_RELEASE_SHA256 = 'eec49c4e92779677ecd87e9c129e0953fd90b5af75c408c216ba356f31544517'
REVIEWED_V22_3_1_REMOVED_MODULES = frozenset(
    'examples/external_reconciliation/' + name
    for name in ('baseline', 'confidence_voxel', 'cross_sections', 'provenance')
)
REVIEWED_V22_3_1_RELEASE_PREDECESSOR_MODULES = {'__init__': {'ast_sha256': 'de03eb2ecb027167e9dfd9d7deeb9ccc8b601207feb50019294605dad11f9e7c',
              'statements_sha256': '0fc64a281e52163c853bf62eb8d767d9babcd1e2eaa9c580e813d9e0a6d8fd31'},
 'cli': {'ast_sha256': '86c31c683fd688cd2259e2f062503e940d1f12867aaef155f23dead8ad88ea14',
         'statements_sha256': 'c6545c14e3606d1b6fb08300b079088c5c9319fa288de3379e43dedc3535eef1'},
 'confidence_evidence': {'ast_sha256': '62d5b21eda07b3e0f8336c0b89eb5fba30ade52dfd8c51560287b567206c5669',
                         'statements_sha256': 'e94647697f14d68d29a6a8514e1e9f560125d6a3de11bb1e779005f2ea854267'},
 'confidence_export': {'ast_sha256': None,
                       'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'confidence_native': {'ast_sha256': None,
                       'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'confidence_projection': {'ast_sha256': '19eeb18873384fe02688f88c338f682be6bea2b9e3478a89e7851ea2765d411b',
                           'statements_sha256': 'bf0c5170a9341db9cb6be7eb21280083955c402106c3f2f5f6f38cd40374157e'},
 'confidence_storage': {'ast_sha256': None,
                        'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'confidence_tiles': {'ast_sha256': '4af149e2c0f4d38a7c40d2a75ceea48b9bbd67b0d307f846af51fb407ddb37d2',
                      'statements_sha256': 'ca0dfdd7a703687256089be0c8fcc367d6398858f63db266654001f64dae9107'},
 'config': {'ast_sha256': '499e31f052868c7830ae01e076a1af2f386b8a9a7535f1a61ca5a92a638bf0f1',
            'statements_sha256': 'a717a9337c3778a87d67ef909bc29d17d2902325fdbae52b61277bfe6acf2440'},
 'cuda_backend': {'ast_sha256': '160b1ae6cc74fc3ed4cbfced2d28abe305af220e30766b3f52ad0a271710198c',
                  'statements_sha256': '77b8f31a4a185035b5e9bc28bc462ffdb2743722451dbcba1fc531c3d121855c'},
 'cuda_d1': {'ast_sha256': '35c03de6d5939061d3c11fd0434306c39bf0c210fb088d2f07a7d7a153c28d6d',
             'statements_sha256': 'a693251016163187b14911f473e6b0a7ade3090f4d00e129928f79f9bf49bca3'},
 'examples/external_reconciliation/_confidence_core': {'ast_sha256': None,
                                                       'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/_hybrid': {'ast_sha256': None,
                                              'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/baseline': {'ast_sha256': 'd0777f6975b4b026908d1126ca48edbba5a05dec629bd570e1d0753334de8ce5',
                                               'statements_sha256': '07c73f1f43fa99220f314f33ae8afce8012507829f50b98909f068c1d14ff281'},
 'examples/external_reconciliation/confidence_core_rescue': {'ast_sha256': None,
                                                             'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/confidence_voxel': {'ast_sha256': '35db2625b59a6edc22b7c0f952409ed5187d6a8005408d154fa1a3e83deb6c41',
                                                       'statements_sha256': 'e149a268eb90fc02ba16e60c50aa0f9e68560f3b505b076f4bf264f52ea29c03'},
 'examples/external_reconciliation/cross_sections': {'ast_sha256': '794269169c537ed32053d22fe8c00d40f4e26ad699502a2a562649b730f2a004',
                                                     'statements_sha256': '18f1ef61f63fca226076671e2c5c98ed12b7a646bda66f1bd0d9a9d743f3eb34'},
 'examples/external_reconciliation/hybrid_with_fill': {'ast_sha256': None,
                                                       'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/provenance': {'ast_sha256': '4f9124ea026670b44b65549248af050e9e9b1ff85980e969d065861a7b864101',
                                                 'statements_sha256': '570628c57aa8debde9ededc8beb982916156cb44f1d481256ee3dbd6384c394a'},
 'examples/external_reconciliation/quorum3': {'ast_sha256': None,
                                              'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'packed_publication': {'ast_sha256': '753a5e91c5c20a37ee042449e204702f45f1d6dd051f4a597060ea2ad799f297',
                        'statements_sha256': '724c3d46ff4ca4fe2562bc108cab0c17435eeb93ba8371d823d0a62f4d5e4457'},
 'pipeline': {'ast_sha256': 'd30f353ccab1960e8f6720fa337aedd86228164859db95d292dd6079510cf905',
              'statements_sha256': '53bc9bcb1c158b09f34a823684d133a5e6d753cbeb549f925da430653c4f124b'},
 'reconciliation_runtime': {'ast_sha256': '0d1e014432ed7199c1252dba5acf74e1ca8d2e26460645c7fd029fe4ee53846c',
                            'statements_sha256': '61f8c8f6301afd2071b63b85bdfe304e80a22eadc0fac530a8a51151b59153d7'}}

REVIEWED_V22_3_RELEASE_PREDECESSOR_COMMIT = 'd336a66d75d7811a2c07d007ed40c46a2cfcfcb2'
REVIEWED_V22_3_RELEASE_PREDECESSOR_SHA256 = '0ff675bc92f9d13147c0218ac87220bc67462ef690bcd986d609e827d1dcd3cc'
REVIEWED_V22_3_RELEASE_SHA256 = '4941424abe38b58144d10692c2706dbd5785669c5183258ac817c0b5d9625ff1'
REVIEWED_V22_3_RELEASE_PREDECESSOR_MODULES = {'__init__': {'ast_sha256': '04dd92c09983d5ce16372d4962ca93578bbc4edcc72070556be74991b24e3ab3',
              'statements_sha256': '02f4266ef8e858f6c816052b871092b86764f689fac4e031c9c800ff71061ee3'},
 'assembly': {'ast_sha256': 'b89fc1c2fd3182e68042b99d36f712865cc9e3a2914bd9405471056eb9f3b284',
              'statements_sha256': '9f4cb561927477b029dcf7a0e0c093d535d4a19e39bca133314f5e8f208b356f'},
 'backprojection': {'ast_sha256': '3d80564861982f1cbe717542d4e2ab50bb162a80cce86fdc08690fe93c7135d9',
                    'statements_sha256': '846f1336d1fa1687ccaa6bd86b22770c503c62e9a14af6c8c89f60b80a52503a'},
 'cli': {'ast_sha256': '9e2b9a79f6f2a40d42cfad38574904fff10cae7bafac5be996b5e443cba12c9d',
         'statements_sha256': '90fdd423a20645fd01f45bce3784ef2d91df9207f953c566665d95d446e533ba'},
 'confidence_evidence': {'ast_sha256': None,
                         'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'confidence_projection': {'ast_sha256': None,
                           'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'confidence_tiles': {'ast_sha256': None,
                      'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'config': {'ast_sha256': '8d60de18764a9b75031667f77a96deec0bf1ae71b16269b04ef6ad513389c6b9',
            'statements_sha256': 'c72b0adf031492bb8900661a19f39a528c25e0b66455700f41c82b9b0465e068'},
 'cuda_d1': {'ast_sha256': '8386813b3e4753d1d1749df437e924b646722f0085347c6095ac714ceae3f564',
             'statements_sha256': 'd0a9e0008c4d61aa4da41c410b1739f3fbfb17aae4f7238ac17489e862cc63cf'},
 'cylindrical_projection': {'ast_sha256': '3b47a0a1a7e649872ef765dccd5a2de03f7e15645ba610a8879b9fe26f2e37f5',
                            'statements_sha256': '9b07c4954627ff253a3e668651dc29220bd39b9a689daee39a9a0d7591293e1c'},
 'examples/external_reconciliation/__init__': {'ast_sha256': None,
                                               'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/baseline': {'ast_sha256': None,
                                               'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/confidence_anchored': {'ast_sha256': None,
                                                          'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/confidence_voxel': {'ast_sha256': None,
                                                       'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/cross_sections': {'ast_sha256': None,
                                                     'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/largest_island': {'ast_sha256': None,
                                                     'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/provenance': {'ast_sha256': None,
                                                 'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'examples/external_reconciliation/union': {'ast_sha256': None,
                                            'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'inference': {'ast_sha256': '21f9319e536ff6aa0b3eec012714764c49b2671833d34c42fc6c337b180c9b5a',
               'statements_sha256': 'f77d44466ecdf3dc961c36482dcc389fdbaf251b2ce069f27ecca1c3ae19dead'},
 'outputs': {'ast_sha256': '887f4c7c8f290baf384d748845e8cea442c487036545ecbcaf601c14bcc69e7b',
             'statements_sha256': '2383024f54ea340a5d15a4b854b4b0c6a0ea49ae10e86ff5a6b845e396b25f3f'},
 'pipeline': {'ast_sha256': '1b0dbd62c3d98c60a6d007f1c2ded7fffce4272a27937e3be0e3d612a01dae12',
              'statements_sha256': 'e0269c3cffcd6de53c07339250c4c3c04902524764fd75bbde47826f1e12c1f2'},
 'publication_memory': {'ast_sha256': '837b997298f7afb5db1476af0d691df21f45e1f43f7abf36d5ac4d7ea1c71c32',
                        'statements_sha256': 'f753d9da219b1fd1b39db60055ba4bf28cee9402c4734aba92fd6b386a15bff7'},
 'reconciliation': {'ast_sha256': None,
                    'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'reconciliation_components': {'ast_sha256': None,
                               'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'reconciliation_geometry': {'ast_sha256': None,
                             'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'reconciliation_io': {'ast_sha256': None,
                       'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'reconciliation_policy': {'ast_sha256': None,
                           'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'reconciliation_runtime': {'ast_sha256': None,
                            'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'spherical_projection': {'ast_sha256': 'd972a635e826951b075f759e5fae41f212e71965fa53c5ffe5ddb1e1c0c509d9',
                          'statements_sha256': '04dc14bb5116dac0abcfeafc1fb44e61e53e9ff5c8148057611843bb458a69b7'},
 'spherical_projection_cpu': {'ast_sha256': '9ee22f42f1303c299baf41db3b274c3270792d8e724da796f99a052a1ee9b3c2',
                              'statements_sha256': '760fd6bd19c409118ec45b2733bb0301ae92a597f8aced1686ff8cbbf908fd95'},
 'tta_scheduler': {'ast_sha256': 'b866e35f83805581baf549a05e03d84468169d6645d1df34794b04eaa6243334',
                   'statements_sha256': '9bf5b0c7435be1d3968a71e1c77f36ed906c966b60ad16c1637172b846781ace'},
 'workers': {'ast_sha256': '51ca45e5343cd464a7ebe1a7229422b3820abc90e62a09ae284f8a85cfdffc22',
             'statements_sha256': '4156111e4999e424ed389997d7ef73e471446a8d403b9d861373f1b7b7893690'}}

REVIEWED_V22_2_RELEASE_PREDECESSOR_COMMIT = 'd99f15a4f9dcedf008324f510227518afdfabb98'
REVIEWED_V22_2_RELEASE_PREDECESSOR_SHA256 = 'e92e36913284b7bcb943f431686ddf19b8fdb7ce2c7a7b9f870e9fddbc3139e6'
REVIEWED_V22_2_RELEASE_SHA256 = '1416dac4e248383f9b435bd1f239bb7459b11479b9f634897d8b9951a96e478c'
REVIEWED_V22_2_RELEASE_PREDECESSOR_MODULES = {'__init__': {'ast_sha256': '49b0f93d2d008e26577dd0861af7d46d428ad754ea160d0ebd1a2c67bdfffbbc',
              'statements_sha256': '09fd3acc4c567013469713f7f00b768886036251cf0bd64e3da3b9cf852d0c4d'},
 'augmentation_policy': {'ast_sha256': '7356ad89169ea3b964b0d27855ea476ee9972cb07690493196d05de343756455',
                         'statements_sha256': '2f2f8bea063bf504a9954be11c2f04b6e7773098cedb201357cdf64158f1fd6b'},
 'cli': {'ast_sha256': '61ce9bd3f6adb54757c56963a2b005eab7938dea5b1395f0ffa66f7b49c819e6',
         'statements_sha256': '73ad0a0f996d64195cb4ebb0092d88d0dc787cd7c934d360e875dfee57637453'},
 'config': {'ast_sha256': 'efe89fa3a8d0cbdec27e7702c399cdd2f30a107e22585791e5e387c9f256fdfb',
            'statements_sha256': '499492f8e1f9f0afdea8ad1aa1812877cad5778d5fadd059f5c16697efaa3726'},
 'examples/external_augmentations/CPU_baseline': {'ast_sha256': 'e87ba1f5276588c25d6a0ed78a42ec2d49325e96ff6ad0e321952368cdb3e2de',
                                                  'statements_sha256': '5b7d12a2749aa3998107ff2a3cbb416e05ea52aac70cbd963fc9e2c51ea0e317'},
 'examples/external_augmentations/CPU_heavy': {'ast_sha256': '028f5a981a60c18622a052aaa4d70a545483f74ae9773483420113b27030bf87',
                                               'statements_sha256': '08fc93297f0bdc5a6e37157e3cdbf621d4ae7497b97fe9610bb509f77806a807'},
 'examples/external_augmentations/CPU_light': {'ast_sha256': '62317b58687a0906e0ef7847b238192c390cbd148b75c78c54eea31fcfbbc6c3',
                                               'statements_sha256': 'b1d53d9792bb8044b51275619d24cee2b40c148f4c29be61872d748161008861'},
 'examples/external_augmentations/CPU_superheavy': {'ast_sha256': '20f97f8c3b2b17fda03439c54827d899fad48006d5b58ea5836ef9dce8df7854',
                                                    'statements_sha256': '2b0b2333519d5d39b9aa2dde13f4d0053ad11142187b3cb74249d314691e8273'},
 'examples/external_augmentations/GPU_baseline': {'ast_sha256': 'ce9e8eeb3628390d2fb7483c1f7698405d35559fa84e5b7bfde4c43af6111047',
                                                  'statements_sha256': '910d8dde13162e2ca48a83130eea7dc72b39ac3a1d6ef7cdba9b089bea66cc4c'},
 'examples/external_augmentations/GPU_heavy': {'ast_sha256': '1a947be443b0aa31b7b632b6438ebbeb3fdcb319e9a8a6d68cad852f2deecb9e',
                                               'statements_sha256': '77cf04c640376f16d139546d68cab7feab866985aa3120d705d4024fc7fe467c'},
 'examples/external_augmentations/GPU_light': {'ast_sha256': 'eee3749d6df7428d027350be17819b4f63417c142c207639ef15cf09b6aab0fd',
                                               'statements_sha256': '4dfdfff3467c65a9191c55dbdc14918720c652dc8231763393be7a9c9f470097'},
 'examples/external_augmentations/GPU_superheavy': {'ast_sha256': 'db8cdea6e1d957570d003778cc81efc9aae0551ba79fb29fe6c547ca8a2db568',
                                                    'statements_sha256': 'b1557d9567784f8740b0337003391150cd6025ba1878b0ebd30054d84f4489bd'},
 'inference': {'ast_sha256': 'f952a27354fad779fdda196b2cd27574195f37ca13b79a965a1700be0564b3b7',
               'statements_sha256': '0beb844471f0eaf54acfca9a7358f236c2602620af8a11ff983d0932da8bd06d'},
 'pipeline': {'ast_sha256': '9caa104ff64303ee7947682dfc5f11138d8031498105c0dd4330a30cadb455be',
              'statements_sha256': 'bc92f7b37e222b938e6fa57a1ba6cfd376e2b611abed6d564fad364fd16e9eef'},
 'pta': {'ast_sha256': '306870f75e2fbb04ba4ec3ffe24cd6e349384f995c209b2934cfc2207c59dbbd',
         'statements_sha256': 'c1277c7bde3d30e30967e4e0b26cb030350b0124705f1d9bf827bd4b35cc948c'},
 'pta_binary': {'ast_sha256': None,
                'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'pta_config': {'ast_sha256': '780c3d3c7197ce596f3f37c5d93929a4db2708cfc1ce6d7189a93f85a879ab3a',
                'statements_sha256': '39af475048faa403993e37135c37e58675698308b05ba52ef5472ada3bad258e'},
 'pta_publication': {'ast_sha256': '56d4265a22ef60319aca1b11a298e3cbcf3941729163012861cbf757e61c9f89',
                     'statements_sha256': '92a5019361dac2b81aa59f096aa50dae79fb50b392e11a45e1ef5faeda4e345c'},
 'pta_runtime': {'ast_sha256': 'e55c20b72fb956d1a7476ec05fe4f00143bdac8f8109c5bdafffa1f0ef7126c0',
                 'statements_sha256': '55d3f47dc4db46a1c58701d064107a20921570cbdf7171605d27964ad1a7cc6e'},
 'pta_workers': {'ast_sha256': '4098ce667b09e72d299226262852a1cb2e60cfb5abf987b466e7fafe9dccaf44',
                 'statements_sha256': 'c403aec8ccd34eed36bdccc3d352b0e992eab254370afc2b6b0c05af628f97f7'},
 'publication_memory': {'ast_sha256': '6fcaa55993fee16839df162e2e442ee57508eedb3a4fd82efcd3a516d25a2afb',
                        'statements_sha256': 'b204f78450b48cda7b1c251b99708ae31cd56864d32af21938efb4e2f13838cc'},
 'tta_augmentation': {'ast_sha256': '076ed27d414f9c9c137f06f827a4c2cbc171ba5dad913d2b25b6429aa95042e6',
                      'statements_sha256': '85d2addab47c3d7cbdacd43fc2cb7778e914d939202c068d9cf90868237950b7'},
 'tta_augmentation_config': {'ast_sha256': '2cdb0bf7074f27c07e4423a81bc76cfe18e201b015b9c616f92f19bce4d7fd82',
                             'statements_sha256': '4fb0c1e3b2f1432302722e9070d7c306972f50c0b272fdff26ae82cec4eb0117'},
 'tta_augmentation_cpu': {'ast_sha256': None,
                          'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'tta_augmentation_cpu_runtime': {'ast_sha256': None,
                                  'statements_sha256': '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945'},
 'tta_augmentation_runtime': {'ast_sha256': '2e5530d5911270ecdc1579b0e1842f35d791610a1bab7f3bcfbe352d7e29c490',
                              'statements_sha256': '19304562e2bd8e824855b69af2c454047bb2624c0432bbf9b00682a421bf5ecc'},
 'workers': {'ast_sha256': 'cc749bc1df655694957424885b64e29f5e6ed437019027b998349bc98bdbde99',
             'statements_sha256': 'a81930b38174fd5ed338b0003dfdcb0f70a2afb42485ec076f2c3f657be4de38'}}
REVIEWED_V22_PTA_PREDECESSOR_MODULES = {
    'pta': '18853a78b464a41017f82d8020f4905d3d5d953d16693abad5067f192b00cfe4',
    'pta_scheduler': '067bfa10b01e5820aa949197e5776af8ce968278b561cf57d46eff883fca5ee6',
    'pta_config': '60d4b4ce90e15b752a225cd998827e4e925c8599d79e59de8bf5137f092d707b',
    'pta_runtime': 'e55c20b72fb956d1a7476ec05fe4f00143bdac8f8109c5bdafffa1f0ef7126c0',
    'pta_publication': '75a806be8a9fd460a6c01e822269d08df5a397d959355a35d3f6941a6c86fe50',
    'pta_workers': 'f0b08380e384f14315db9f883f41c7d38bdeaec391a135150a43b0eee1c58f5c',
    'nvtiff_backend': 'eb785f353231a46f5bfe59f8f85b1d77300c1d47f9f3d2a05bdd215ee6572f75',
    'pta_batch_pipeline': None,
    'pta_gpu_publication': None,
}
REVIEWED_V22_PTA_DEFINITION_KEYS = {
    ('nvtiff_backend', 'NvTiffBackend'),
    ('nvtiff_backend', '_cuda_image_device_pointer'),
    ('nvtiff_backend', '_image_info'),
    ('pta', '_planning_phase_affinity'),
    ('pta', 'main'),
    ('pta', 'write_pta_summary'),
    ('pta_batch_pipeline', 'BatchReservation'),
    ('pta_batch_pipeline', 'OrderedBatchPipeline'),
    ('pta_batch_pipeline', '_PendingBatch'),
    ('pta_config', 'build_pta_argparser'),
    ('pta_config', 'parse_output_image_format'),
    ('pta_config', 'parse_pta_args'),
    ('pta_config', 'resolve_pta_config'),
    ('pta_gpu_publication', 'GpuPublicationResources'),
    ('pta_gpu_publication', 'GpuPublicationTask'),
    ('pta_gpu_publication', '_GpuBatch'),
    ('pta_gpu_publication', '_byte_limit'),
    ('pta_gpu_publication', '_finalize_resources'),
    ('pta_gpu_publication', 'publication_resources'),
    ('pta_publication', 'NvjpegCudaFenceError'),
    ('pta_publication', 'NvjpegEncodedBatch'),
    ('pta_publication', '_encode_nvjpeg_batch'),
    ('pta_publication', '_publish_nvjpeg_batch_atomically'),
    ('pta_publication', '_write_nvjpeg_batch_atomically'),
    ('pta_publication', 'parse_output_image_format'),
    ('pta_scheduler', 'PtaCpuBudget'),
    ('pta_scheduler', 'PtaPipelineDepth'),
    ('pta_scheduler', '_bounded_gpu_cpu_sets'),
    ('pta_scheduler', 'plan_pta_cpu_budget'),
    ('pta_scheduler', 'resolve_pta_pipeline_depth'),
    ('pta_workers', '_gpu_runtime_for_worker'),
    ('pta_workers', '_label_payload_bytes'),
    ('pta_workers', '_publish_gpu_policy_batch'),
    ('pta_workers', '_publish_label_payloads'),
    ('pta_workers', '_render_worker_initializer'),
    ('pta_workers', '_validate_gpu_policy_batch'),
    ('pta_workers', '_write_gpu_image_batch'),
    ('pta_workers', 'execute_gpu_frame_batch_task'),
}
REVIEWED_V22_PTA_STATEMENT_KEYS = {
    ('nvtiff_backend', 'binding__NVTIFF_PHOTOMETRIC_RGB'),
    ('nvtiff_backend', 'module_docstring'),
    ('pta', 'import_.pta_scheduler'),
    ('pta', 'import_contextlib'),
    ('pta_batch_pipeline', 'binding_PayloadT'),
    ('pta_batch_pipeline', 'binding_ResultT'),
    ('pta_batch_pipeline', 'import___future__'),
    ('pta_batch_pipeline', 'import_collections'),
    ('pta_batch_pipeline', 'import_concurrent.futures'),
    ('pta_batch_pipeline', 'import_dataclasses'),
    ('pta_batch_pipeline', 'import_operator'),
    ('pta_batch_pipeline', 'import_typing'),
    ('pta_batch_pipeline', 'module_docstring'),
    ('pta_gpu_publication', 'binding__QUARANTINED_PUBLICATIONS'),
    ('pta_gpu_publication', 'import_.pta_batch_pipeline'),
    ('pta_gpu_publication', 'import___future__'),
    ('pta_gpu_publication', 'import_concurrent.futures'),
    ('pta_gpu_publication', 'import_contextlib'),
    ('pta_gpu_publication', 'import_dataclasses'),
    ('pta_gpu_publication', 'import_os'),
    ('pta_gpu_publication', 'import_threading'),
    ('pta_gpu_publication', 'import_time'),
    ('pta_gpu_publication', 'module_docstring'),
    ('pta_publication', 'binding_OUTPUT_IMAGE_FORMATS'),
    ('pta_publication', 'binding__NVJPEG_QUARANTINED_BATCHES'),
    ('pta_publication', 'binding__NVJPEG_QUARANTINE_LOCK'),
    ('pta_publication', 'import_concurrent.futures'),
    ('pta_publication', 'import_dataclasses'),
    ('pta_scheduler', 'binding___all__'),
    ('pta_scheduler', 'import_dataclasses'),
    ('pta_workers', 'import_.pta_publication'),
    ('pta_workers', 'import_contextlib'),
}
# Exact pre-move PTA definitions, independently pinned before their shared owner
# is introduced. PTA reexports must still resolve to the shared owner objects.
REVIEWED_PRESERVED_AUGMENTATION_DEFINITIONS = {
    ('pta_augmentation', 'AugmentationDefinition'):
        '7f4a5e8fdb7a64582a6aa510c4381a2eb5853b030ebd8e3a0a4940346072c9f9',
    ('pta_augmentation', 'inspect_augmentation_definition'):
        '3d9f9bf614d79d46a7a09c72887db6fa7fda6006f7e2dc662aff51bf3ec86ca0',
    ('pta_augmentation', 'assert_augmentation_definition_unchanged'):
        '06e131ad10dbfd062a4e4b2aaca6c370340a00cf43cbf717006b7054b982aa8a',
}
# This method was previously covered by the immutable full Radial module and
# class pins. Name its exact pre-v21.0.6 AST before reviewing the upload change.
REVIEWED_PRESERVED_RADIAL_UPLOAD_SHA256 = '5f12562dafcb991f93b1702b8976e33a5117e4e01de357c3cb99c98afe52599a'

# These definitions have reviewed, intentional implementation changes.
INTENTIONALLY_CHANGED = {
    ("assembly", "materialize_interpolation_component_nrrd_view_layer"),
    ("assembly", "project_view_volume_to_orthogonal_volume"),
    ("backprojection", "_backproject_cartesian_azimuthal_generic"),
    ("backprojection", "_backproject_tilted_azimuthal_volume_to_volume"),
    ("backprojection", "backproject_azimuthal_volume_to_volume"),
    ("cuda_backend", "union_conf_volume_into_volume_inplace"),
    ("cuda_d1", "_d1_finalize_bitset_layer"),
    ("finalization", "_v14_apply_component_removal_plan"),
    ("finalization", "union_volume_into_volume"),
    ("inference", "cleanup_view_volume_after_prediction_inplace"),
    ("inference", "fill_view_volume_holes_2d_inplace"),
    ("inference", "fused_slice_cleanup_inplace"),
    ("interpolation", "IncrementalRawBBoxMaskStoreWriter"),
    ("interpolation", "PreparedViewResult"),
    ("interpolation", "_build_linear_slice_bridge_plan"),
    ("interpolation", "_component_record_mirrored_u"),
    ("interpolation", "_drain_volume_to_mmap"),
    ("interpolation", "_estimate_linear_slice_bridge_min_radius_from_plan"),
    ("interpolation", "_write_raw_bbox_payload_store"),
    ("interpolation", "materialize_raw_bbox_mask_store_workspace"),
    ("interpolation", "write_raw_bbox_mask_store"),
    ("media", "LazyProcessingCube"),
    ("media", "decode_video_to_memmap_gray8"),
    ("media", "resize_categorical_volume_to_processing_cube_uint8"),
    ("media", "resize_volume_t_axis_only_gray8_slab"),
    ("media", "resize_volume_to_processing_cube_gray8"),
    ("media", "restore_mask_volume_to_original_shape"),
    ("media", "should_resize_to_processing_cube"),
    ("outputs", "resize_binary_mask_volume_to_shape"),
    ("outputs", "resize_gray_volume_to_shape"),
    ("pipeline", "_main_impl"),
    ("runtime", "_ensure_process_backed_interpolation_volume"),
    ("runtime", "_mount_fstype_for_path"),
    ("runtime", "close_memmap_array"),
    ("runtime", "open_raw_store_payload_writer"),
    ("topology", "fill_3d_voids_inplace_streaming"),
    ("topology", "_adjacent_gid_pair_codes"),
    ("topology", "label_foreground_volume_streaming"),
    ("backprojection", "_MainProcessGpuStageCoordinator"),
    ("backprojection", "_ResidentTensorRTRingExecutor"),
    ("backprojection", "_azimuthal_resident_backproject_kernel"),
    ("backprojection", "_resident_trt_pipeline_acquire"),
    ("backprojection", "_try_resident_trt_ring_accumulate"),
    ("backprojection", "HybridBackprojectionQueue"),
    ("config", "build_argparser"),
    ("config", "resolve_save_request"),
    ("config", "resolve_backend_batches"),
    ("config", "resolve_backend_precisions"),
    ("cuda_backend", "_GpuWorkerRenderEngine"),
    ("cuda_backend", "_fused_direct_render_kernels"),
    ("cuda_backend", "_azimuthal_slab_channel_renderer"),
    ("cuda_d1", "_d1_backproject_kernels"),
    ("cuda_d1", "_d1_consume_device_union"),
    ("cuda_d1", "_D1WorkerViewState"),
    ("cuda_d1", "_shutdown_d1_worker_pipeline"),
    ("cuda_d1", "_d1_get_or_create_state"),
    ("cuda_d1", "_nrrd_layer_key"),
    ("cuda_d1", "_nrrd_layer_name"),
    ("cuda_d1", "tile_dense_worker_result_limit_bytes"),
    ("cuda_d1", "tile_dense_worker_result_limit_tasks"),
    ("finalization", "apply_keep_largest_objects_inplace"),
    ("geometry", "ChannelFormattedFrameRenderer"),
    ("geometry", "InMemoryYoloVolumeSource"),
    ("geometry", "PredictionVolumeRef"),
    ("geometry", "StreamingYoloVolumeSource"),
    ("geometry", "channel_view_slice_index"),
    ("geometry", "build_view_frame_cache"),
    ("geometry", "dense_tile_positions"),
    ("geometry", "gpu_input_staging_ahead_sources"),
    ("geometry", "get_azimuthal_sampler"),
    ("geometry", "make_dense_tile_channel_renderer"),
    ("geometry", "make_fullframe_channel_renderer"),
    ("geometry", "make_in_memory_yolo_source"),
    ("geometry", "make_prediction_ref_yolo_source"),
    ("geometry", "materialize_dense_tile_prediction_volume_for_job"),
    ("geometry", "materialize_fullframe_prediction_volume_for_job"),
    ("geometry", "maybe_eager_stage_prediction_ref_on_gpu"),
    ("geometry", "queued_streaming_source_cpu_warmup_slots"),
    ("geometry", "render_dense_tile_frame_for_job"),
    ("geometry", "render_fullframe_frame_for_job"),
    ("geometry", "streaming_prediction_source_prefetch_frames"),
    ("geometry", "streaming_prediction_source_workers"),
    ("geometry", "should_cache_view_frames"),
    ("geometry", "tile_crop_border_pixels"),
    ("geometry", "tile_parent_crop_window"),
    ("geometry", "write_aug_job_meta"),
    ("geometry", "write_dense_tile_job_meta"),
    ("geometry", "resolve_tile_configs"),
    ("geometry", "extract_azimuthal_slice_frame"),
    ("inference", "cpu_retina_masks_enabled"),
    ("inference", "PredictionAccumulationHandle"),
    ("inference", "_DeviceUnionAccumulator"),
    ("inference", "_ResidentGpuPipelineSlot"),
    ("inference", "_resident_mask_kernels"),
    ("inference", "_try_create_device_union_accumulator"),
    ("inference", "gpu_union_retirement_lane_count"),
    ("inference", "predict_in_memory_volume_and_accumulate"),
    ("inference", "predict_in_memory_volume_and_submit_accumulation"),
    ("media", "abort_streaming_producers"),
    ("media", "decode_video_to_memmap_gray8_streaming"),
    ("media", "processing_volume_mode"),
    ("media", "resize_volume_to_processing_cube_gray8_streaming"),
    ("media", "_cube_t_axis_resize_backend"),
    ("outputs", "_publish_staged_file_atomically"),
    ("outputs", "_MemberParallelGzipPayloadWriter"),
    ("outputs", "_announce_nrrd_cpu_deflate_backend"),
    ("outputs", "_nrrd_gzip_executor"),
    ("outputs", "_nrrd_member_codec_candidates"),
    ("outputs", "_nrrd_member_codec_self_test"),
    ("outputs", "_nrrd_member_codec_spec"),
    ("outputs", "_open_nrrd_payload_writer"),
    ("outputs", "_select_nrrd_member_codec"),
    ("outputs", "_try_gpu_downbin_volume"),
    ("outputs", "_try_gpu_downbin_volume_on_device"),
    ("outputs", "NrrdLayerSink"),
    ("outputs", "write_binary_tiff_sequence_from_pattern"),
    ("outputs", "write_layer_nrrd_with_low_quality_mirrors"),
    ("outputs", "write_single_layer_nrrd_from_ref"),
    ("outputs", "write_view_images"),
    ("outputs", "write_yolo_labels_from_pattern"),
    ("outputs", "nrrd_gzip_compresslevel"),
    ("outputs", "nrrd_member_codec_requested"),
    ("pipeline", "main"),
    ("runtime", "_GpuWorkerAuxInterpolationPool"),
    ("runtime", "_record_runtime_feature_gauges"),
    ("runtime", "_materialize_worker_task_memfd_paths"),
    ("runtime", "copy_workspace_array"),
    ("runtime", "choose_scratch_dir"),
    ("runtime", "interpolate_view_volume_pass_maybe_process"),
    ("runtime", "interpolation_process_start_method"),
    ("runtime", "reset_runtime_state_for_new_run"),
    ("runtime", "RuntimeTelemetry"),
    ("workers", "run_prediction_volume_in_worker"),
    ("workers", "_gpu_inference_worker_main"),
    ("workers", "_OpenVinoCpuSegmenter"),
    ("workers", "run_prediction_volume_in_openvino_worker"),
    ("topology", "_try_label_slices_stage_a_gpu"),
    ("topology", "build_slice_endpoint_seeds_from_label_volume"),
    ("interpolation", "SliceEndpointSeed"),
    ("interpolation", "NrrdLayerRef"),
    ("interpolation", "SliceBridgeRenderPlan"),
    ("interpolation", "SliceSeedBridgePlanResult"),
    ("interpolation", "_paint_linear_slice_bridge_plan_onto_slice"),
    ("interpolation", "_paste_local_mask_onto_slice"),
    ("interpolation", "_plan_slice_seed_bridges"),
    ("interpolation", "interpolate_view_volume_pass_inplace"),
    ("interpolation", "interpolation_planning_backend_name"),
    ("finalization", "assemble_view_volumes_and_projected_layers_fused"),
    ("finalization", "assemble_current_view_union_volume"),
    ("finalization", "_v1401_embedded_plane_ridges"),
    ("finalization", "_v14_plan_components_and_write_sparse_audits"),
    ("finalization", "_union_projected_layer_ref_into_volume"),
    ("assembly", "finalize_consolidated_tile_volume_for_parent"),
    ("assembly", "gate_tile_residual_against_parent_bridge"),
    ("assembly", "gate_tile_result_against_parent_mask"),
    ("assembly", "materialize_nrrd_view_layer"),
    ("assembly", "_try_apply_gaussian_smoothing_gpu_chunked_inplace"),
    ("assembly", "apply_gaussian_smoothing_inplace"),
    ("assembly", "spill_waiting_tile_result_to_raw_store"),
    ("geometry", "is_tilted_view"),
    ("outputs", "write_summary_file"),
    ("outputs", "nrrd_layer_output_suffix"),
    ("cuda_backend", "GpuRenderedYoloSource"),
    ("cuda_backend", "GpuTileRenderedYoloSource"),
    ("cuda_backend", "_azimuthal_slab_context_indices"),
    ("workspace", "v1613_d1_pipeline_active"),
    ("workspace", "v1613_fast_bundle_active"),
    ("workspace", "available_anon_work_bytes"),
}

# The public wrapper owns the full-run cleanup boundary and delegates to this private
# implementation name.
INTENTIONALLY_RENAMED_CHANGED = {
    ("pipeline", "main"): "_main_impl",
}

INTENTIONALLY_VERSIONED = {
    ("config", "451b35336c86c625bd71b77e55c8a09bef571c75405977484e6e0e6debadcd51"):
        "SCRIPT_VERSION",
    ("config", "bbaeec59e08232583950d10ce19229b162f82f5f41ec60afe2dff19fc2e9c6b2"):
        "SCRIPT_VERSION_COMPACT",
    ("config", "7eeed39e30c270fc4e56bbef52e6bc94b6e61bce6599988450ac920bb180a67f"):
        "SCRIPT_BASENAME",
}

# Non-definition bindings whose reviewed contract changed after the immutable baseline.
# Pin the original statement digest and require the named replacement binding to remain
# unique, matching the version-binding treatment without misclassifying it as metadata.
INTENTIONALLY_CHANGED_BINDINGS = {
    ("config", "9a8d538aa3d7fa8f8d2cf55e46f6ac5b31ff4bc6b5823d7bf242954e9055c6df"):
        "SAVE_OPTION_TOKENS",
    ("geometry", "a4f438f50fb19a43076e30f5f4b09acf5b68ca6487f501f2c51f2e2b4bd86623"):
        "_AZIMUTHAL_SAMPLER_CACHE",
}

# Functions that need to call back into a higher architectural layer carry this marker
# immediately above an explicit function-local import.  Treat that narrow import seam as
# a reviewed AST change without weakening statement coverage for the function body.
LOCAL_IMPORT_SEAM_MARKER = "# Local import keeps the package dependency graph acyclic."

# Each entry pins both the complete reviewed top-level definition and the exact marker-to-
# import associations inside it.  The second digest covers the import's relative offset,
# enclosing lexical scopes, and normalized ImportFrom AST.  Comments are absent from Python's
# AST, so pinning only the definition digest would still let a marker move to a different
# already-existing local import without review.
REVIEWED_LOCAL_IMPORT_SEAMS = {
    ('assembly', 'prepare_view_volume_after_fullframe'): (
        'd52de78c1ba5464c0fb4b0234b96c9c0cdf430e93100974defce364d55001721',
        'c9595db954da98f5df4678a2bbf2416a70dc432e38c05097ed3396e6d0c823f6',
    ),
    ('assembly', 'finalize_consolidated_tile_volume_for_parent'): (
        'bc3ecd1d7d9f2d9e9f158a290d0e075bae85565b2b299cfe84f08b08ec7491a4',
        'c955712cc1202c0be55b529d57026f25537c1d43e0e9c692ad6cdc85b3b913f5',
    ),
    ('backprojection', 'backproject_tilted_volume_to_volume'): (
        'ae645bd53d368d0216171d90afe0bef10b7b1dd1dee80499b0eca6f0c8d49a9c',
        '101a6fb2b4446cf0d71be4e025f1224cfecb863185f21e3482bfcfc9cf3a3231',
    ),
    ('cuda_backend', '_GpuWorkerRenderEngine'): (
        '369e578af8e515cc079c738f58d91518a06a6e834e1f55d74533dffad1ea4d35',
        'e25d3d3f292164aee026e8e9c8d49caf37a2be6241c7c36f65aca52d31d3bbf5',
    ),
    ('geometry', 'GpuPrefetchingYoloSource'): (
        '1234b0ac1454e2d643e3688a7b94aab0510961d19ff4b0c91d4720f000568c17',
        'b230c59c54aa3f8c1c0efe9ffd6c4d6172f49531e530ce0a02acfc252c19a35d',
    ),
    ('geometry', 'gpu_input_staging_enabled'): (
        '3ac4bb523c36af4f98daf48f3153a3813cc9459f2aa879846459c4c3e3352e70',
        '27c0b26fbbf4d1ccfa8a5af862a09cce91b81ea32d281a1b5301c84d4eb2876e',
    ),
    ('geometry', 'gpu_input_staging_preflight_reserve'): (
        '0fab02a75013aaff85c7b63328c518f3c2022e2de9fe8c5f64471b2a4cdde919',
        '0dab677f37a32eea1b3a0138dcf73c6b876b41093fee8d61af77c89abbeb9699',
    ),
    ('geometry', 'maybe_wrap_source_with_gpu_input_staging'): (
        '496bf1060982040fb935f932af7624277ab7f4c8f8abe4f01f7864f1ab1f4213',
        '28dda3f8902c87b048214bf8d3dd28f3117785ea42503b66e20db91056c6c2e2',
    ),
    ('geometry', 'ensure_ultralytics_accepts_in_memory_volume_source'): (
        '60badc92d2dc2b6667dd10e4d14364d82fb8496c819c2e6a523682eca0802030',
        '857b70aaccd5a89c0104cd8a7bb39fea7e92d30adc84025789fc7f03cfc81eb4',
    ),
    ('geometry', '_materialize_prediction_volume_from_renderer'): (
        '134a732f44c0e24d20a4cd2bd779bb48b291088a4f56d0d71d009dc909de9da9',
        '4a8cb9169fcfbc4bf1fa1658615d8f3ae1377e8ac32cc1aeaa65707f37e22cd3',
    ),
    ('inference', 'infer_yolo_model_input_channels'): (
        '7e8d329f12766affd78ee938e593bb989e8535d157c375b143f4fbbeee1bec6c',
        'd0e25f9ae060e0c7d74bece86ae3befb49f781d4a703061bb311fcb2c8d4f410',
    ),
    ('inference', 'predict_source_and_accumulate'): (
        '28eec86038df26fe016c533232272e809cf340c62a8acc83a262fac84d190308',
        'ccf37e9656817910658d67d9c07dd114ce6eb29a820fda1bfa904ac246d955d8',
    ),
    ('inference', 'predict_source_and_submit_accumulation'): (
        '025d1c3210be00b360d318306e71fcef8211d7ce116a43a77ca46a42dcead5eb',
        '1ce8d95596c7f073801ec32a1245bb5751595f1b3d6b5a2884063a126b89efdf',
    ),
    ('interpolation', 'SliceComponentTableCache'): (
        'fcb31853f671ae2dc0a8a7e9bada7e481186cd2b2d5c3369f803e850ff9bceab',
        '29f4bad74e6bec994ef8fa59ab290dcb079b0a5ae918342c108b472e66ab66ab',
    ),
    ('interpolation', '_find_slice_projection_candidates_numba'): (
        'cbeebf1855fd025de032082a83f62ed77b1120e6e16bdeba779c4f0badfc8e29',
        '7fbd6e38c3db9d3821d9622ed891b1b7fbfe0f838c27de910d0944c749cfdcda',
    ),
    ('interpolation', '_find_slice_projection_candidates_python'): (
        '2964a4a06f74b43fbdca643ab9663fd3332fa8f854a2ae69f3fae162a7775dc0',
        '94740a539572d09297cb90016159014fc849ececb6d5c529e605783041542bfd',
    ),
    ('interpolation', '_build_slice_endpoint_seeds'): (
        'bf29b06e72824fa78871fe502eb7ea79057bd352138dbf90bda55ca62f0d30ae',
        '126545d0d25722c4df5918428643130e3c6a1eb639a0c61476fd7a259ad57cdf',
    ),
    ('interpolation', 'interpolate_view_volume_pass_inplace'): (
        '0c91e9acd48ec2d9b7470e9b8143b0a77329ac59ecb01428288055595934e2b6',
        '0a585dbad86412327820dccb21e479e86057fa65bb0ae01a016b198238a3f661',
    ),
    ('interpolation', 'RawBBoxMaskStore'): (
        '5e076700cb529dfbbb7a0e6bb7f7b582249b312252ab7510afe9da701402aa0a',
        'a19c94672d63a393b3a647ec73bba9da4791667f7b0082fa9f739ee56c50d16c',
    ),
    ('media', 'resolve_azimuthal_azimuth_angles'): (
        'be46c0979af4d4395c6538c789df7ca43659c4c5f66b01175d0e9fafb7117087',
        '94f8c4ba510aa3756118ce2ae981a3ce25f3d6eebc81c2b096045efe004b6b12',
    ),
    ('runtime', 'gpu_worker_default_seconds_per_frame'): (
        'cbdc743efad682f4c852ac135af8a38a9c3a85103499dfdedd36f92dbe0618d6',
        '8417466843f56b5afeaeef3a2d20fd67a91a90b0355438581bb5b2eadb1f7623',
    ),
    ('runtime', 'gpu_worker_task_cost_key'): (
        'f98176aac67c05de805593dccc266ab9e6f595c2989a180d638cb2948f05f552',
        '96f2a949e6948895eba3a583fa9f3da197a532674e991b8f98b921a196bff5ea',
    ),
    ('runtime', 'cpu_inference_supports_view'): (
        'ef761cc4da5ee4ba113200b9985d925659139b97d4d8c67201f0c8ffb989aa2a',
        '2175b73531245efd35fd7b3ffef54a31b2eba271bb82f19d9a1f710796bd3c2f',
    ),
    ('runtime', 'cpu_inference_task_priority'): (
        'cff4f59a9287337a997965bbfb9a63ebc1c1c1bd252cf11318317736d347c144',
        '50a97e164a4f6a098fbf7e773de161a7a1ad4a8ce05c9c37f286606a965df3bb',
    ),
    ('runtime', '_interpolation_process_entry'): (
        '83c1a7c9b379a384e2d285231ec33dbc595870b09e793b62be7fc8baaf0bff91',
        'babab2eb1d231e00bab18c5e3d624b34d92401f4a0fadd98b03035b6d38e0775',
    ),
    ('runtime', 'interpolate_view_volume_pass_maybe_process'): (
        '00604305e8c8b8fab50881a1bc8a3e41b9265bd7a3d58515619527622e499f92',
        'fa8ba3a3af40efa0605c8b9e41e422281719adee141eeb6376b2e589eb88b213',
    ),
    ('topology', '_try_label_slices_stage_a_gpu'): (
        '6dce9807e442982e580688a8466cb4076b7cf5d22db2599d943ac51b915aceff',
        '79f6cfde9d6e366582897499240cf9ea21081d92328d16c658d9113891e0004e',
    ),
}

INTENTIONALLY_RELOCATED = {
    ("runtime", "70e22341666e8e63ad2a0a0239676cd85eb4aba6e256a55379ec7958cbc35799"):
        "workers",
    ("runtime", "a6ecc26570ba0d1a5feda101d96bfa587013159b1bf5370b5875aff9ed3ff212"):
        "workers",
    ("inference", "bf2ffc53f405ceff8a38bc5f846d09608a4ac82575a58d8938ceb0204d8bc99b"):
        "backprojection",
    ("inference", "fcc88a83030aa0dac3b506d345ee748d01c4cdc9e0e3b08aa9549cbfde0d44ca"):
        "backprojection",
}

# Keep the baseline inventory intact and account for each retired statement by its baseline
# digest, so adding a similarly named definition later cannot silently satisfy this audit.
INTENTIONALLY_REMOVED = {
    ("interpolation", "3e5d65dbd592dcec65808a55789c42f16ae2685e2992c890415c1a049c8dd124"):
        "_component_record_to_local_canvas",
    ("interpolation", "512225d0daf2f7d71f236c27fae5effa7c0a410f1175366cabc0a1987718843d"):
        "_local_half_width_for_component_records",
    ("interpolation", "7a2a71d0dec9df268cbf9efe76cf84c6812c1c0d0ad5e5856838ae7a52196314"):
        "_component_to_local_canvas",
    ("interpolation", "bcf33c955845b6f1e36dcc21bb95ade865cd980493a2e2631b7e0245c8ab94ea"):
        "_local_half_width_for_components",
    ("runtime", "6f3557199f1b478a7f72c087dd15e2d28bdf5a49b6b4a8ff29707dc038b2cfb6"):
        "raw_store_memfd_enabled",
    ("runtime", "49fc4622c58ba272d947cb15e4790c1eca1ca8f3f0bed161ba4ef678137ee088"):
        "_create_memfd_backed_payload_path",
    ("runtime", "1bebc854b4c96d9f0f827c2d5df1b735fcd1ce404fe82841a2787ce01af879a5"):
        "flush_array",
    ("runtime", "1fcce8668dc1d71e4ea52d70cd57a9ab19eb0a97fc27b474cdf722be725b6a02"):
        "prediction_volume_build_flush_enabled",
    ("runtime", "dd953b74f4d66f1146464d5faa74a8e3ce2d664e7b7890f9071c9f6e21a9b003"):
        "prediction_hot_path_flush_enabled",
    ("config", "0b77703bf375bcd802f74a77ca9009db17a87296bb829376ed2f30f368250243"):
        "OUTPUT_NRRD_PREFIX",
    ("config", "4b5b1cd71ab26699413794dc1b3b0a1b1a7b91dbb917f0321bd4fda5fe2b95d8"):
        "LEGACY_OUTPUT_NRRD_PREFIX",
    ("config", "479a50756ba923fbe000d52aad5dea92511533a044b7903224de6715fd9301a7"):
        "variant_nrrd_stem",
    ("config", "e9cbdca394845cae9bdb26ad2d5cdfd5dea831d31b29c8b305e97300328760ee"):
        "AZIMUTHAL_TEXTURE_VARIANT_LABEL",
    ("config", "813fa551257393b30cd4587ca2fdfa2de5cf42351be75a57f94c0e03f0b210ca"):
        "resolve_save_options",
    ("config", "071ba93675e9d91466da964da542560b322ea80c4498c011939bd24ebacca524"):
        "_parse_quantize_arg",
    ("finalization", "357c81e4223f2c6b6fd247ce2c44088ad228499def98cce9d3ab8ae430bfa5a0"):
        "assemble_views_concurrency",
    ("finalization", "7e3ee2b874386829aab5aa902979c279a4b412b6e835f3c55d92aba377867431"):
        "assemble_view_volume_from_projected_layers",
    ("geometry", "f294bcd6122f87aa1128cb47877d0a2761738d2b50fc91e304a6491d5afd17f0"):
        "tile_jobs_uniform_crop_shape",
    ("inference", "beac6387c6688fec98b2fc6023e8f034b6489e34bae549240bcbc139e24938eb"):
        "background_model_load_enabled",
    ("inference", "65033f136d739e98455671add94f47d951b82b708e7a02629c68916e5301deec"):
        "_canonical_single_device_token",
    ("inference", "820c3099703d78a02772d1e785c98008df67ecc0e18073787f43b43a8f0df783"):
        "parse_device_list",
    ("inference", "34e6bdedc1de53c986f30028efc59349d36b7d872296ca5fe7ad0c3b655d629a"):
        "is_cpu_device_list",
    ("inference", "7ec809ae495bc5b969690a8cbd4193a63cfb192b26a83dbc113229d7101b427f"):
        "resolve_retina_mask_processor",
    ("interpolation", "0f8e158500f0a81ec31578174c31fde27e072bcc4b4f54718f63f88a7d3b2f62"):
        "_component_records_directly_overlap",
    ("outputs", "bed6cab37b1b3c47b34c2204852d8e6d5d77d787c56193da32a91587b72dfc75"):
        "_NRRD_GZIP_EXECUTOR",
    ("runtime", "4ae29b5a9b0c7626a2e0f2e4daefd6fc1f9cb3d37a1d44888dc14e563306a144"):
        "scratch_shm_required_free_bytes",
    ("runtime", "2c5358a6d7546d61f2bb33b5bfe44152266dd35372e50c366636f611f221948d"):
        "_auto_shm_scratch_candidate",
}

# The compiled overlap helper lived inside a larger top-level conditional.  Pin both the
# baseline inventory digest and the reviewed replacement digest so the verifier still
# authenticates every sibling kernel in that statement after the one dead helper is pruned.
INTENTIONALLY_PRUNED_REPLACEMENTS = {
    ("interpolation", "e2a3ab6f0b2bb8abfd8cb880317a197a446bd80d52fd49f7c4bc72608dbfe529"):
        (
            "4e01060dae0bb88826bf2b113bcb0f5e9d10f2ade407861176dd712774bb430e",
            "_numba_blocks_overlap_any_kernel",
        ),
}

# v20 additions to formerly preserved definitions are pinned individually. These
# are semantic changes, never accepted by the mechanical Azimuthal rename map.
# Key: immutable (module, baseline AST digest). Value: current name/digest/reason.
REVIEWED_V20_STATEMENT_REPLACEMENTS = {
    ('interpolation', '7f10eb5c476c42205a2aa865d3465c1ebabcad7a5a7d2a7042f4bd0d00c27187'): (
        '_raw_store_chunks_cache_key',
        'b8948f64f81c29198edc73b608230a46a89e633957a2074ae88ddf6d0334fd30',
        'Keep logical layer-path cache identities distinct across equally named memfds and stable through backing-descriptor retirement; retain per-layer shared mmap refcounts.',
    ),

    ('outputs', '9a9119faff78922a2fb2f826ef0ee290c9612194bb6ca984490a77420bede185'): (
        '_write_one_decomposed_nrrd_layer_payload',
        'c5ff52a5c1982a4eed513b1c35a01fa9fd385313515ac03049a394ab39291984',
        'Stream native bbox row bands and cached zero spans only for software member writers with no dense-block observer; retain restored, dense-observer and hardware routes.',
    ),

    ('cuda_d1', 'bb0b1ecf37d4c52647a15defe499c22667663ca95fb7de4ec6cfbf0021143061'): (
        '_d1_submit_publication',
        '5c92d44692a1f8910a8b8e332c06102828089a552de95881345de07d23d7f4d8',
        'Reuse bounded source-bitset publication with explicit native-shell provenance while preserving legacy output semantics.',
    ),
    ('geometry', 'c8e5256bd662acf22cc3768a96593fce8084353e0c54c2bb00f80c64d771075a'): (
        'ViewInfo',
        '698023677c765317066783bf66b23a0d618d2cc37b315ffe8148255c0124306a',
        'Distinct shell trajectory fields retain azimuthal metadata and provide explicit radius/patch coordinates.',
    ),
    ('geometry', '7032646b8c76f03c753731ba4ae6b573a3b2c999679b88c411b778834c1c0992'): (
        'get_view_infos',
        'f77b27e9a095317082509cb8bd88f5a98dd561d3818ae74a351340ed9511c116',
        'Compile shell requests after unchanged existing view-family order.',
    ),
    ('geometry', '7971d05259012d2615747c9fb8af8bd5e7454f1a354624341bf0d32fc37dad82'): (
        'get_view_frame_by_index',
        '9459aacb1f573c141b78224b6ca3be8596dabe93050f9e74574e04143cfca6d0',
        'Dispatch new radial trajectories through bounded native shell rendering after source readiness.',
    ),
    ('cuda_backend', 'c8972e4acbdde22a8fe221daae7d343900a8d4862710528e4643374ca19093b7'): (
        '_fused_preflight_family',
        '7543d6727bbe5978c32a0e2f978f9560c83beb092fa334e6b9b98aacd687f6b4',
        'Exclude shell views from fused kernels that only understand Cartesian and angular planes.',
    ),
    ('finalization', 'fd81cf9fc0a5587f79c19bbb789f2ee537872fa0bbbbeeda884375ee660b75c3'): (
        '_v14_sample_one_normal_section',
        '9f05a7abde1d13bd2fd201fd4f7cebd2c830c089178e6deac61901083a8347b4',
        'Use the physical radial_distance variable name; square-root and annulus arithmetic are unchanged.',
    ),
    ('interpolation', '225ccfafc2b01273762daff87a7b9e9d1b1aecaa5d29a0d1e65aabbc617822b8'): (
        '_view_uses_interpolation',
        'bef22390c1d652d4e07a88261acd5ea5d16685f4c11517e7aad08e87e46e2ff5',
        'Allow interpolation within independent radial radius stacks.',
    ),
    ('cuda_d1', 'c476ceeb33c77a59a0833f212f1a773eaf632c11819693121f46eb762742ebbc'): (
        '_d1_view_family_ids',
        'e9c75f246b0b9b6102d6758f0c917ef9ea0de1a1a997bfd7716b047e74ff43df',
        'Reject shell views at the incompatible D1 kernel boundary; native union projection handles them.',
    ),
    ('inference', '58b9a7e4a8df5eba9cd5a011380d1baae25576338d2ecad05669ba8f3882a571'): (
        '_build_direct_device_compacted_payload',
        '01022e9cde7573fbd0883364c2aefde85cce82dffa0020cb4962cdb815f02324',
        'Match the generic compact CUDA kernel eight-argument signature with image height, width and null optional bounding boxes; required by shell inference.',
    ),
    ('geometry', 'e0203b6d1f34c8f3463b5a344c6e7b40a22d6fe2ba67552bf9323b20ea9b59f9'): (
        'view_output_token',
        'd4b65eef0239cccad875fc7906131975aea307b033653b6d982c4a864a00002c',
        'Give upright and tilted shell patches distinct filename-safe Radial output tokens retaining patch indices.',
    ),
    ('runtime', '105ed98db9ab4f6929a3244915a10ed1a3c14104f15068420e44bb9c89fa5b50'): (
        '_sched_setaffinity_all_threads',
        '0f11d5b4d48f7454adcb92d28c0cd2872e50481f0a1556c33f3405783afec7f4',
        'Dispatch Windows to a verified process/thread affinity implementation with rollback; preserve the Linux affinity implementation.',
    ),
    ('inference', '9931df03e8cd8b9142d3ab7cae0b896ed054b3648fd9700fa2fe7149fe2be16b'): (
        '_split_segmentation_backend_outputs',
        '1ca4b7224262330ee4bc332ca2f16571a87fdb0abbde91321edb17ce60a38ce7',
        'Accept the measured current Ultralytics PyTorch ((head, prototype), auxiliary dict) segmentation layout while retaining flat exported and legacy tuple outputs; reject unsupported tensor dimensions.',
    ),
    ('backprojection', '5530061ab3afe471577e68c8c7466e0147b9da5ac845fe5f93cb8ddfe16fff69'): (
        '_ResidentTensorRTRingExecutor',
        '3645e9c5a49b9fd9f74c3fa325f4d9d6f2ba872bbbb6d7de10f9d5a848179041',
        'Retain idle TensorRT ring buffers across radial generic tasks; fence context handoffs, restore/rebind tensor addresses, and discard/recapture the changed borrowed-context inference graph before reuse.',
    ),
    ('backprojection', '9b514b1cfc81798ca0b8ac84760457aa09df110e5294af349e7b9743c8871671'): (
        '_resident_trt_pipeline_acquire',
        'f13530361829bad0e0f79b7db8cd47cd838b002bd97837b30a9ef12f6da87a14',
        'Resume retained ring contexts and track source renderer/volume identity before reuse.',
    ),
    ('backprojection', '7a65755d524b14876ed83f1b64ebcfabe777ba3690f855b09c84e267a5924509'): (
        '_try_resident_trt_ring_accumulate',
        'a19f4eb0d7c6dc7395e792abd2934b67d9ecc27f8a35aba35d8a75e0763c9d08',
        'Soft-suspend compatible radial generic tasks and run all-view fused preflight only after single-channel static TensorRT admission, before borrowing context bindings.',
    ),
    ('workers', 'f6c091cc8da972d22cd13d9135cc862ecbcc92b2c9ccca540d48d8a7b6eeb794'): (
        'run_prediction_volume_in_worker',
        '6dd4ac73111d24f25196e3442378ab5a434eb44dee84c6fb4fc88cdfff6fad2b',
        'Avoid irrelevant eager fused preflight for generic paths and emit flushed per-family render/result provenance, including CPU fallback.',
    ),
    ('assembly', 'fbe3c9b679863c1c1e49085f5e3b98ca730c8bc2f6d96979ed390a7a2b190a05'): (
        'project_view_volume_to_orthogonal_volume',
        '75a39a1f829c04ed66d55c9da77ebcfc41df8a5bd12c1690874f16c100e5975e',
        'Forward existing exact per-shell mask bounding boxes into the factored Radial projector without changing other view dispatch.',
    ),
    ('pipeline', 'a2c4a36ea39c43fb451b6667bdb8aa0507d56f849d4d7b3768dd46476af98fc2'): (
        '_main_impl',
        '60496d3e8cc73cceb8d0608c33dfb7c6510671379bfd07fab7cf70ddba47e6a4',
        'Authorize skipped non-interpolated drain copies only when component-ref retirement is active, tiling is absent and temporary artifacts are not retained; existing terminal ownership retires the original.',
    ),
    ('assembly', '43e838c80f5d1abeb35369a50161eb5363f95cfaa560d9b6f7cf437dc1c20567'): (
        'materialize_nrrd_view_layer',
        '487f1f3d7a8af4456500621fb6e18ccf2b791967b93851267b12b23d0eeb4f81',
        'Fatal projection failures abort/discard incomplete CPU sink stores and raw scratch without retry, including fatal failures during a transactional dense retry; preserve the original fatal exception and quarantined GPU owners.',
    ),
    ('interpolation', '7780e8aa5a241f9942c17c57e60f3acb446da4da3fb7396dd8bd01ddcd146b17'): (
        'IncrementalRawBBoxMaskStoreWriter',
        'fc3eeba9edd8e3982f4cc2c7ac90f94a0d5d8f80ddaab98acef7f24d3a3cba4c',
        'Validate complete encoded raw/packed record addressing and padding before atomic reservation; append owned payload synchronously without rescanning pixels, preserve failure invalidation, and provide serialized positional-write fallback on Windows.',
    ),
}


# New v20 helpers are authenticated separately from historical accounting.
# Kernel source is a string constant inside its factory's AST, so its digest
# covers the actual CUDA arithmetic as well as Python compilation/fallback logic.
REVIEWED_V20_ADDED_DEFINITIONS = {
    ('interpolation', '_raw_store_chunks_cache_key'): (
        'b8948f64f81c29198edc73b608230a46a89e633957a2074ae88ddf6d0334fd30',
        'Keep logical layer-path cache identities distinct across equally named memfds and stable through backing-descriptor retirement; retain per-layer shared mmap refcounts.',
    ),

    ('nrrd_spans', 'canonical_zero_member'): (
        'c1c7188458aa910ccf03590a1f551763090a650b3465d63629cdf5b843e664a5',
        'Encode native bbox row spans and cached compact zero members without changing decoded bytes, sparse observers, hardware codec policy or failure ownership.',
    ),
    ('nrrd_spans', 'stream_native_crop_spans'): (
        '9eeda37617a222f06413c78a3d15d45c1a0040876ded2da1cec031d12fc84011',
        'Encode native bbox row spans and cached compact zero members without changing decoded bytes, sparse observers, hardware codec policy or failure ownership.',
    ),
    ('outputs', '_MemberParallelGzipPayloadWriter'): (
        'f18fc29b7bb1c13a73e8cbc8260ca228561bdc6c611447347cc1ab9df8426d7d',
        'Encode native bbox row spans and cached compact zero members without changing decoded bytes, sparse observers, hardware codec policy or failure ownership.',
    ),
    ('outputs', '_write_one_decomposed_nrrd_layer_payload'): (
        'c5ff52a5c1982a4eed513b1c35a01fa9fd385313515ac03049a394ab39291984',
        'Encode native bbox row spans and cached compact zero members without changing decoded bytes, sparse observers, hardware codec policy or failure ownership.',
    ),
    ('publication_memory', 'publication_output_reserve'): (
        'de3daa4868896f867ad2a826766d692c70e92ddfe1b5a71c29d29af0a8c408e1',
        'Reserve actual gzip windows, mirror canvases and global compressed spools before retained RAM admission.',
    ),

    ('cuda_d1', '_d1_get_or_create_state'): (
        '37cff0c7bc9c3e636a5fe68f5bc75fd180835667ab091371dce8a132db5ce6cf',
        'Publish exact packed source crops without dense expansion; carry bounded parent RAM grants and preserve NumPy/raw fallbacks.',
    ),
    ('interpolation', 'IncrementalRawBBoxMaskStoreWriter'): (
        'fc3eeba9edd8e3982f4cc2c7ac90f94a0d5d8f80ddaab98acef7f24d3a3cba4c',
        'Support parent-owned RAM payloads with private prepublication disk spill, bounded copying and binary descriptor writes.',
    ),
    ('packed_publication', 'PackedOwnerCrop'): (
        'a23fcab310937aecd6e0bd5a10bbcf42efb49da674ac56c1ff1faf217a56b8ac',
        'Direct source-bitset bbox/count and row-packbits encoding with exact addressing, padding and bounded output blocks.',
    ),
    ('packed_publication', 'encode_owner_packed_block'): (
        '5584fce731c199ad90439018360b8f346a04ac6f528e68a4552973a62b498d0d',
        'Direct source-bitset bbox/count and row-packbits encoding with exact addressing, padding and bounded output blocks.',
    ),
    ('publication_memory', 'plan_native_publication_memory'): (
        '3fc6e04c0fee91391d33b66d6f0f5f500894044d93740899513f7938a0c98420',
        'Pre-dispatch worst-case retained-layer admission, physical/cgroup headroom, future-work reserve, parent descriptor lifetime and disk spill policy.',
    ),
    ('publication_memory', 'publication_ram_headroom'): (
        'c019bc77c877709c2106e25116cb8cc47c3ff255ce4f53613f232dc268fc4ea3',
        'Pre-dispatch worst-case retained-layer admission, physical/cgroup headroom, future-work reserve, parent descriptor lifetime and disk spill policy.',
    ),
    ('publication_memory', 'retained_payload_plan'): (
        '7a06a971b4815a9fbc9777cae369dc78d724206725e677297ece64bd8cd76555',
        'Pre-dispatch worst-case retained-layer admission, physical/cgroup headroom, future-work reserve, parent descriptor lifetime and disk spill policy.',
    ),
    ('topology', '_compiled_adjacent_gid_pair_codes'): (
        '5f6698213b323cc9b2cfa6b4ae4846b328efc7d0ac17cf71267e11b8a32f80df',
        'Use exact bounded row-run intersections before the retained pixel-hash adjacency fallback.',
    ),
    ('topology_runs', '_run_adjacent_pair_codes'): (
        '68093630d4d287bc4d316feb8b30cefdd752275b812588f37b60091546fe3b1c',
        'Bounded equal-label run intersection preserves all touching pairs and falls back on fragmentation or optional compiler failure.',
    ),
    ('topology_runs', 'run_adjacent_pair_codes'): (
        '2e3bf434f38c21856f794382ea355c18a76a39580bd070f422401a469a8e8619',
        'Bounded equal-label run intersection preserves all touching pairs and falls back on fragmentation or optional compiler failure.',
    ),

    ('cylindrical_owner', '_bucket_shell_pixels_numpy'): (
        'f247ed177d854a52dad612649a85d5ee3f19d81f427e8eb19b2acb9edb586f68',
        'Stable vectorized shell buckets preserve exact pixel order when optional JIT compilation is unavailable.',
    ),
    ('cylindrical_owner', 'shutdown_radial_owners'): (
        '934300c210658758a74c761fa460398ee226e7b4c684e774afddb57c4cec08db',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'preflight_radial_owner'): (
        'ddb926a9b0dbd536691f1bf5ac15f321a1225a03625cfbd8d11f2d368648b5db',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'active_radial_owners'): (
        '24f66e6f282c5e1990c14a45130c63e815ecb418062bc96b3dd698c5f7e436a3',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'consume_radial_device_union'): (
        'd6a6f408781926417ded5fe2eb809e94fb9d2a23fe7dfe41dcbb1f26c38e9b88',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'RadialOwner'): (
        '710eb39c851132e4d7865a03c5a6bebb4f0b2dd10b84d17dd39277620e1a4042',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', '_bucket_shell_pixels'): (
        '234cadc5b2e81fcaacd479e6c8054f9e82243bb4e69e6f0cec30bb83ee19e1e1',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'is_radial_owner_task'): (
        'ef6221af1b8063ddd35778f54a66801ed2fc1c5abd0617554c84d75d8fe34755',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'radial_owner_eligible'): (
        '4b3759b7a8f350f77087dec613f5d21b13cb5bb7d7a633d86cd743bb80b19478',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'radial_runtime_provenance'): (
        'f84efc29da511d5b2cad028396a7fb1aca93973c381393de1c96b14862ae4c16',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'radial_owner_enabled'): (
        '37fe66f517ab6dff9ff2d6a1f60ac12b3ed1865bb406ce7eb58416a3bc5eabb3',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('cylindrical_owner', 'DeviceOnlyRadialTarget'): (
        'cf86365ca84e4b8aad65f863ae1f87477a637df9cc142e45a657cad8ac799801',
        'Native-shell owner boundary: exact cleanup/projection, coverage, bounded resource ownership or runtime provenance.',
    ),
    ('tta_scheduler', 'TtaScheduler'): (
        '596f4ca180ab0586b60731ff0a6e04d7f3a905ddf6c6d5198d284d57a4f9110d',
        'Keep native radial tasks on one owner even when the multi-owner experiment is requested.',
    ),
    ('cuda_d1', '_d1_submit_publication'): (
        '5c92d44692a1f8910a8b8e332c06102828089a552de95881345de07d23d7f4d8',
        'Reuse bounded source-bitset publication with explicit native-shell provenance while preserving legacy output semantics.',
    ),
    ('cuda_d1', '_d1_finalize_bitset_layer'): (
        '44ef5a4b81c1630ca48eee683bf10622e49b8cade7ebedff33db8daf867c0a43',
        'Reuse bounded source-bitset publication with explicit native-shell provenance while preserving legacy output semantics.',
    ),
    ('inference', '_prediction_accumulation_target'): (
        '9adaefe94493deba1ad8a04d0d7a191f167507576ccb48a307aab18cf7cba858',
        'Support shape-only native device results, reject duplicate/missing radius results and avoid host cleanup on device-only frames.',
    ),
    ('inference', 'predict_source_and_accumulate'): (
        '28eec86038df26fe016c533232272e809cf340c62a8acc83a262fac84d190308',
        'Support shape-only native device results, reject duplicate/missing radius results and avoid host cleanup on device-only frames.',
    ),
    ('workers', '_gpu_inference_worker_main'): (
        'fe54c592cd78e1e67dedca3f736a934900a77bb95531421191c2d32731b0b0bb',
        'Dispatch native shell consumers without old D1 geometry/proto cleanup; retain native-owner retirement barriers and fatal CUDA ownership.',
    ),
    ('workers', '_release_gpu_worker_inference_assets'): (
        'cc82e02acb0e19f80792e32e4b516c0063ea28b706d3e2b1562cb01456816a7d',
        'Dispatch native shell consumers without old D1 geometry/proto cleanup; retain native-owner retirement barriers and fatal CUDA ownership.',
    ),
    ('workers', 'run_prediction_volume_in_worker'): (
        '6dd4ac73111d24f25196e3442378ab5a434eb44dee84c6fb4fc88cdfff6fad2b',
        'Dispatch native shell consumers without old D1 geometry/proto cleanup; retain native-owner retirement barriers and fatal CUDA ownership.',
    ),
    ('pipeline', '_main_impl'): (
        '60496d3e8cc73cceb8d0608c33dfb7c6510671379bfd07fab7cf70ddba47e6a4',
        'Route eligible shell views through the native owner contract, preserve compatibility and empty-layer policy, and log effective execution provenance.',
    ),
    ('cylindrical_cuda_projection', '_pack_radial_source_block'): (
        '8af8b8387c7a92773486ef7eb7decdae0f8df0bb960f65c678490055e2129b4f',
        'Copy exact cropped uint8 source intervals across shells and row boundaries using validated integer addresses and bounded output ownership.',
    ),
    ('cylindrical_projection', '_build_radial_plane_plan_reference'): (
        'c6d7097429247bd42b40ebee82506a068f2cd934f2aeba7e930671f9fa31c557',
        'Retain the original two-pass CSR assembly as an exact numerical reference for qualification.',
    ),
    ('cuda_backend', 'radial_native_kernel_enabled'): (
        '5fb5888625044f859e03b198d29f1f06f1180ecf2c6b7fb3c617f3dc6fbda0ad',
        'Explicit opt-out for the scalar native radial CUDA renderer; default enablement retains a logged Torch fallback.',
    ),
    ('cuda_backend', '_radial_native_kernels'): (
        'd452ff9b1a55bb4ce7d57da2830fb3c6570601e71b7b018bd992e23113e2e5bd',
        'Compile and cache the bounded scalar-geometry radial CUDA sampler; preserve float32 accumulation/uint8 rounding, deferred logical-T values and zero-extended taps.',
    ),
    ('backprojection', '_resident_trt_pipeline_suspend_for_radial'): (
        'fb076a9d5e579a9900ee47e7fbe50e126fb1c28b55ac12e7f2869c1cb196d03f',
        'Suspend only idle matching model/source radial ring contexts; mismatches retain full invalidation and active/failed handoffs fail closed.',
    ),
    ('workers', '_announce_worker_render_path'): (
        'd587d682581704f2d325c323d443c89884b941955efeb2b3c85b2224c90f33a2',
        'Emit one flushed render/result route record per family and task-kind combination in each worker.',
    ),
    ('cylindrical_projection', '_RadialPlanePlanTooLarge'): (
        '0df1909707cf449838365cfd7cd18ecc011e63b0194b2147fb5124397f2612e3',
        'Explicit bounded-plan admission exception selects the unchanged reference before any output is delivered.',
    ),
    ('cylindrical_projection', 'RadialPlanePlan'): (
        'f0ef083a324833515f2888a4cecf76bb6cabf61d225b1287ec007530c7d1ea1a',
        'Readonly 2D shell ownership and CSR periodic-column plan with explicit resident byte accounting.',
    ),
    ('cylindrical_projection', 'clear_radial_plane_plan_cache'): (
        'fcffb209a2ab00bb2137946a1d00fc33f0aa22c76f483659a2a5f6a3222ab54e',
        'Advance the cache generation and detach active build Futures without cancellation; old callers complete without repopulating or removing newer-generation entries.',
    ),
    ('cylindrical_projection', '_plane_geometry'): (
        'a795e43d7e1dea439d1e80c025602c89901cdf8d40a816c10593988f75936e43',
        'Resolve source/processing base-plane axes independently of the varying stack axis.',
    ),
    ('cylindrical_projection', '_plane_occurrence_strips'): (
        '98efa5553f4e6440cdfbc50e4b95ff54b17b0e982b576dfd0bcab950942aeb63',
        'Preserve exact NumPy nearest-shell and repeated-addition wrap equations while allowing bounded single-pass construction strips.',
    ),
    ('cylindrical_projection', '_build_radial_plane_plan'): (
        'a54394857ccfe6a62e40206623e76317615a627814483152dd7a4a5f58abd63e',
        'Assemble exact readonly CSR tables in one geometry pass; bound strip occurrence retention by the plan budget and preserve per-pixel wrap order.',
    ),
    ('cylindrical_projection', '_radial_plane_plan'): (
        '4733fa3e98aea57e1fef235f2da26754ef44228a60af9f5e3552dea7acc26144',
        'Share one Future per concurrent exact plane key, complete outside the cache lock, preserve original build errors and retry eligibility, keep unrelated keys parallel, and retain readonly bounded LRU ownership.',
    ),
    ('cylindrical_projection', '_radial_projection_metadata'): (
        'f961fe2aab01cd9ab17778436351c78cbec4176bf881afc8f963235b5d94f8af',
        'Evaluate the original NumPy sampled-shear equations in bounded multi-shell batches; preserve exact processing maps, ideal shear, float64 rounding and tilt signs.',
    ),
    ('cylindrical_projection', '_gather_radial_pixel'): (
        '865c22ec8d83336bc596b6dae0dce1ea08d4f9277d3fee27e9707382cbe18e5a',
        'Gather nearest mapped mask pixels with original height ties and periodic OR; valid caller-supplied bounds only skip known-zero reads.',
    ),
    ('cylindrical_projection', '_project_radial_block'): (
        'acc24813912d8759bca03f78fc7b3903e15c0e61ba2615ea77110305aa5d0601',
        'Apply the same factored pixel gather across each axis orientation in bounded source-output blocks.',
    ),
    ('cylindrical_projection', '_radial_block_schedule'): (
        '00299ca499d5d419dc25f26762132750698cd4e38ced577d5fe02ad58363892f',
        'Bound output block depth and concurrency by requested/allocated CPUs and the explicit in-flight byte budget.',
    ),
    ('cylindrical_projection', '_ordered_radial_blocks'): (
        'd6c2d1e36d7ff36676f1e6e97f6cf3a15d3ed422f7c4edc2b848985aba3adcca',
        'Deliver concurrent projection blocks in source order, cancelling/draining remaining work before borrowed-input retirement on failure.',
    ),
    ('cylindrical_projection', 'backproject_radial_volume_to_volume'): (
        'ef799fa756fe233a89229c297b8a078fb57a37a83f3c42de9b7270cdd3a8b74b',
        'Preserve direct-CUDA production dispatch and setup partitions; report opt-in graph and empty-packet experiments separately without enabling them at the pipeline boundary.',
    ),
    ('cylindrical_projection', '_nearest_global_shell'): (
        'cb42c5d91613549ba8ad3fd2d644ed2f4e7ff67ebfd526371b00c162266cd07f',
        'Unchanged v20.0.1 reference nearest-global-shell ownership, including inner-shell midpoint ties.',
    ),
    ('cylindrical_projection', '_processing_index'): (
        '0370c22c421c00617b6434802420e7b57029cfad1c21d21513b2125d4a9c55b1',
        'Unchanged v20.0.1 nearest processing-mask index mapping.',
    ),
    ('cylindrical_projection', '_occurrence_rows'): (
        'a30e06657361bc2dfd8686b405ca0810177a9565bf6628833c3e95226f964ea8',
        'Unchanged v20.0.1 per-occurrence inverse-shear reference and global height rounding.',
    ),
    ('cylindrical_projection', '_pull_radial_chunk'): (
        '0425fba0d265589fd0500e2a865b17dc623a5e74a19dc7563173137015d8d09f',
        'Unchanged v20.0.1 NumPy scalar-grid reference retained as the independent oracle and oversized-plan fallback.',
    ),
    ('cylindrical_projection', 'radial_cuda_backproject_enabled'): (
        '3a1b0fba08221b9f0796c968770c7d99739fba94d6bf1fbf695ec9498967d32c',
        'Explicit opt-out for Radial CUDA projection while retaining the unchanged CPU projector.',
    ),
    ('cylindrical_projection', '_RadialCudaStage'): (
        '559197c333d917c55321816e00d2bc8ea48840f86b7978bba2f450fb86ef6b64',
        'Hold the device lease across dense or encoded block production and consumption; forward encoded format explicitly and preserve unsafe fence quarantine.',
    ),
    ('cylindrical_projection', '_close_radial_cuda_resources'): (
        'c7101d75a1756a95a86bf12666620e66d4f8049c068eca0f3cea4bd3c4fe7894',
        'Fence and release resources uniformly for normal exits and constructor/wrapper failure; unsafe fences preserve projector and lease instead of re-admitting the device.',
    ),
    ('cylindrical_projection', '_try_radial_cuda_stage'): (
        '1ed7bea44466397b00deb2e6f945f395e90b802201fd936904740bd709549772',
        'Respect ordinary inference-priority/retirement admission; fallback to CPU only after safe prepublication construction cleanup.',
    ),
    ('cylindrical_projection', '_ordered_radial_cuda_blocks'): (
        'aa0f49b0efda3841a9f9bd47860d2230ce982d5045c31dfd87ca7dba89e4accf',
        'Prefetch dense or encoded blocks with one GPU producer and independent host ownership; settle pending work before cleanup on consumer failure.',
    ),
    ('cylindrical_cuda_projection', 'RadialCudaProjectionUnavailable'): (
        '327436e46a1747c7d221807e7e58762cf0cdc7d21743d5f023ac5b9bb2db08b0',
        'Recoverable prepublication CUDA admission failure distinct from unsafe device ownership.',
    ),
    ('cylindrical_cuda_projection', 'RadialCudaProjectionUnsafeFailure'): (
        'e1225e7a8ab44ffbab8fe12d069404123f8f47fb92fda547b5917e3afd9eb7a5',
        'Fatal BaseException retains the unfenced projector and bypasses ordinary retry handlers.',
    ),
    ('cylindrical_cuda_projection', '_ProjectionContract'): (
        'b7dd54823355c0e35f7680495de00c4ab99f2d644b9b57a54964c65a13facf55',
        'Validate dense or bbox-cropped source addressing under the existing output and geometry admission contract.',
    ),
    ('cylindrical_cuda_projection', '_positive_shape'): (
        'f2e346180fb107cbfb7ec1cba507825c83caac4571aa7ea38c0b9d49e587a341',
        'Require integer positive int32 dimensions before CUDA address construction.',
    ),
    ('cylindrical_cuda_projection', '_contract_array'): (
        'bbae841e3f206f9ddd72ca2dc60e917ef515a8c403330fe98283ab351ce33772',
        'Require exact contiguous dtype/shape for every borrowed host lookup array.',
    ),
    ('cylindrical_cuda_projection', '_validate_projection_contract'): (
        '2954b1598a06fbb4184bcc86f24b147aff63bc6dd592712b1d44854251c330d1',
        'Prefix known source rectangles with uint64 offsets and select cropped addressing only when it reduces storage; preserve gather validation.',
    ),
    ('cylindrical_cuda_projection', 'RadialCudaProjector'): (
        '123acc79a54c42266679f2ac69c175ab22a4f4884b47f5b83c10129cc073ca29',
        'Keep graph replay and empty-packet elision disabled by default after mixed throughput results; explicit experiments retain bounded capture, preflight, direct fallback and fatal ownership fencing.',
    ),
    ('cylindrical_cuda_projection', 'RadialEncodedSlice'): (
        'bc52167c2887163a857a68a9bb873901c53ba29312558d3d56100868717620e4',
        'Immutable per-slice crop bounds, exact foreground count and payload offset/size metadata.',
    ),
    ('cylindrical_cuda_projection', 'RadialEncodedBlock'): (
        '471701be9affe1da43e554088140a728865b7c5fed88b637d3ef13272ed1987f',
        'Immutable encoded block identity/format with independently owned readonly concatenated host payload.',
    ),
    ('cylindrical_cuda_projection', '_encoded_records'): (
        '8b984daf20133cf42ad805c476925031d5b1c87fb3bb7cac81eb9ddbfc52d4e8',
        'Validate device crop bounds/counts, normalize empty slices, and prefix bounded raw or little-endian row-packbits payload offsets before device encoding.',
    ),
}

# Compile policy and bounded resource constants live outside function bodies.
# Pin their complete AST statements so helper hashes cannot mask policy drift.
REVIEWED_V20_ADDED_STATEMENTS = {
    ('packed_publication', 'optional_compiled_kernels'): (
        '832ee0081e592b680682d6a7182c671420ed965717ed8c673a35bdd0b2932bd5',
        'Direct source-bitset bbox/count and row-packbits encoding with exact addressing, padding and bounded output blocks.',
    ),
    ('topology_runs', '_TOPOLOGY_RUNS_DISABLED'): (
        '914bd52d8607bfd7d187e0e84a1b98fe50e5de2536576ee006ee079f564eaf71',
        'Bounded equal-label run intersection preserves all touching pairs and falls back on fragmentation or optional compiler failure.',
    ),
    ('topology_runs', 'optional_compiled_kernels'): (
        '6824f9609888b53198dac49de1d9ce860e0894d8c36a2d16c926c7a75a872ddc',
        'Bounded equal-label run intersection preserves all touching pairs and falls back on fragmentation or optional compiler failure.',
    ),

    ('cylindrical_owner', '_OWNER_KERNEL_SOURCE'): (
        '4d833c54ff172cfc150730b1ed57f574aac648051198623231368939af4b0877',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', '_bucket_shell_pixels_compiled'): (
        '8ba550259e851dc5379807dd87dc1867f613202d431e039a280d854829d7229e',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', '_RADIAL_OWNER_LOCK'): (
        '22f5cde9003cdd04ef1057fe0f92b1552cb53764c95cece589a11564fdc97f93',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', '_RADIAL_OWNER_STATES'): (
        '23eb5b965d00c50f4f6e2868d42a6a8b605baa7f148c68c43491ff5908393cd9',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', '_RESERVE_BYTES'): (
        'b1dcdc89cbca2b1e25e13f67a7844a474b129eb0d81ea3bc53806dd1a1745fb2',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', '_WORK_ITEMS'): (
        '1f26b2c99d35da871b02c17234d80d5bea31cf753604b75ff4cb45b2ea8cfcf9',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_owner', 'RADIAL_OWNER_CONTRACT'): (
        '07d4d26a31de759bf9b189f01f331f1e1a946b405ca62c2d0d0c73814cedd024',
        'Pin the native owner protocol, kernel, optional compile policy and resource limits.',
    ),
    ('cylindrical_projection', '_RADIAL_METADATA_CHUNK_VALUES'): (
        'cf19116408f9aa4dc22e7eb1308055641a96ec0eebb2df7ed17a3ceb5c9d8e7c',
        'Bound sampled-shear intermediate arrays at one million values per batch, with one native row as the minimum.',
    ),
    ('cylindrical_cuda_projection', '_pack_radial_source_block_compiled'): (
        '765802ea54a2b542140955b926906800d9fc54561b7e564f210855851ecb3746',
        'Keep source crop packing optional, cached and nogil; compilation failure before upload retains the NumPy path.',
    ),
    ('cylindrical_projection', '_PLANE_BUILD_CHUNK_PIXELS'): (
        '4afc438f9808e94475fae6bf58c44b4e2b9c6177bd0761924ba3ca533dba3a0c',
        'Bound single-pass NumPy coordinate strips at one million pixels to reduce interpreter handoffs during concurrent output work.',
    ),
    ('cylindrical_projection', 'numba_compile_policy'): (
        'a00fd59c3a0ef57a76f093e34ec03927219de47f7f0ad44fe7c22b21bc713347',
        'Numba compilation retains cache=True/nogil=True/fastmath=False and inlines the exact pixel gather; no relaxed floating-point reassociation.',
    ),
    ('cylindrical_projection', '_PULL_CHUNK_VOXELS'): (
        '92570f47d7a6bc732f8519a8171b4a369d8b4565c378b819e7f6e4d37f913191',
        'Explicit bounded projection/plan resource policy; authenticate values outside function bodies.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_CACHE_BYTES'): (
        '06c9fc845d716743a26e1300257b98422676643431557777b3df8345cb237591',
        'Explicit bounded projection/plan resource policy; authenticate values outside function bodies.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_MAX_BYTES'): (
        '074d4ff4b7a21a7b54aca28e3418535238ba6ab5807b15c33cb93009c94e842c',
        'Explicit bounded projection/plan resource policy; authenticate values outside function bodies.',
    ),
    ('cylindrical_projection', '_OUTPUT_BLOCK_BYTES'): (
        'bf8f6f6165a78206c9c57c5d280b4154c1a81a74f2ef70db2c1bb8318887c002',
        'Explicit bounded projection/plan resource policy; authenticate values outside function bodies.',
    ),
    ('cylindrical_projection', '_INFLIGHT_OUTPUT_BYTES'): (
        '61a8484d4af12b9ffcdea6c7d29178336c7c1e93c5e590adefff6db39892b91c',
        'Explicit bounded projection/plan resource policy; authenticate values outside function bodies.',
    ),
    ('cylindrical_cuda_projection', '_BLOCK_BYTES'): (
        '2f39654da65e61d24adab7838ee405a85d095361e889e4e73826b29fb7f02d16',
        'Explicit bounded private CUDA/pinned staging and admission reserve policy.',
    ),
    ('cylindrical_cuda_projection', '_UPLOAD_BYTES'): (
        'e40900fa2a6f0fc6a68b8f654bf86342ab930f214b05a494c8bdaadff982f81c',
        'Explicit bounded private CUDA/pinned staging and admission reserve policy.',
    ),
    ('cylindrical_cuda_projection', '_RESERVE_BYTES'): (
        'b1dcdc89cbca2b1e25e13f67a7844a474b129eb0d81ea3bc53806dd1a1745fb2',
        'Explicit bounded private CUDA/pinned staging and admission reserve policy.',
    ),
    ('cylindrical_cuda_projection', '_SETUP_BYTES'): (
        'c06d5b317b175243352d3c3519e3ddfa7e0ae74e090087d86086836d213d8aec',
        'Explicit bounded private CUDA/pinned staging and admission reserve policy.',
    ),
    ('cylindrical_cuda_projection', '_KERNEL_SOURCE'): (
        '87efaa993cf5c91cd6999f39f3f8938b19b1d1f89bc2e4bb91a0cd9337a51244',
        'Preserve exact cropped/dense source addressing and float64 geometry; captured projection optionally reads first-Z from an owned device scalar without changing gather math.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_CACHE'): (
        'dda54f3a989503868661b93cc95518d3728061aae4a0c40bcf152e5fa990d316',
        'Initial bounded LRU and generation-scoped shared-build state with one common lock.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_CACHE_SIZE'): (
        '17894c39f5d5d8acfa6c95744f1fcac6d246e182cd559f8d5f63cd0fde3bfa1d',
        'Initial bounded LRU and generation-scoped shared-build state with one common lock.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_INFLIGHT'): (
        '581530db643eaabdcf4f9772aec89497a9b01a927a43cdefb7274dfaa815bfb7',
        'Initial bounded LRU and generation-scoped shared-build state with one common lock.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_CACHE_GENERATION'): (
        '10e839a0d670a2f9adff107705216624be23d07018b4448754806389f1d3b808',
        'Initial bounded LRU and generation-scoped shared-build state with one common lock.',
    ),
    ('cylindrical_projection', '_PLANE_PLAN_LOCK'): (
        'bda5024f36dde3b97e50ccb2aee349e0decab14bccec493d2401329d3854ff7b',
        'Initial bounded LRU and generation-scoped shared-build state with one common lock.',
    ),
    ('cylindrical_cuda_projection', '_MAX_ENCODED_SLICES'): (
        'c5a08a0217c3c75befe725137cbc60a52a3fcd01f6bc6c7a8ae7be45f2673a01',
        'Bound compact metadata to 4096 slices with the exact aligned 24-byte crop/count device ABI.',
    ),
    ('cylindrical_cuda_projection', '_CROP_METADATA_DTYPE'): (
        '49ee0788784009e994d9a3be35539f969e78601744e0590b4568c789d36c9e62',
        'Bound compact metadata to 4096 slices with the exact aligned 24-byte crop/count device ABI.',
    ),
    ('cylindrical_projection', 'shared_future_import'): (
        '7785747cb9647444f6bf0c7aae1df3a2a18fc2b09299d488ca78744db5dfecc7',
        'Use the standard concurrent Future and executor for shared plan-build ownership and ordered producers.',
    ),
    ('interpolation', 'encoded_operator_import'): (
        '60e489325a19bee8be6a450945dd253634f01297b803d2869e489219288b5372',
        'Use integer-index validation for device-encoded records before reserving writer state.',
    ),
}


_AST_DUMP_SHOW_EMPTY = 'show_empty' in inspect.signature(ast.dump).parameters
_ACTIVE_DIGEST_CACHE: ContextVar[dict[ast.AST, str] | None] = ContextVar(
    'inventory_ast_digest_cache', default=None)
_ACTIVE_REVIEW_AUTH_CACHE: ContextVar[set[tuple[object, ...]] | None] = ContextVar(
    'inventory_review_auth_cache', default=None)


def stable_ast_dump(node: ast.AST) -> str:
    """Serialize an AST without Python 3.13's default empty-field elision."""
    if _AST_DUMP_SHOW_EMPTY:
        return ast.dump(node, annotate_fields=True, include_attributes=False,
                        show_empty=True)
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def digest(node: ast.AST) -> str:
    cache = _ACTIVE_DIGEST_CACHE.get()
    if cache is not None:
        cached = cache.get(node)
        if cached is not None:
            return cached
    normalized = stable_ast_dump(node)
    value = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if cache is not None:
        cache[node] = value
    return value


def current_baseline_name(name: object) -> object:
    """Map historical declaration names only; never normalize current source ASTs.

    The old Radial family became Azimuthal in v20. New Radial shell definitions
    must not satisfy those historical statements merely by reusing their names.
    """
    if not isinstance(name, str):
        return name
    return name.replace('radial', 'azimuthal').replace('Radial', 'Azimuthal').replace('RADIAL', 'AZIMUTHAL')


def azimuthal_rename_replacements(manifest: dict[str, object]) -> dict[tuple[str, str], str]:
    """Validate exact baseline-to-rename AST hash pairs, retaining all original hashes."""
    statements = manifest['statements']
    baseline = {(str(item['module']), str(item['sha256'])): item for item in statements}
    review = manifest.get('v20_azimuthal_rename', {})
    replacements = {}
    for item in review.get('statements', []):
        key = (str(item['module']), str(item['baseline_sha256']))
        if key not in baseline:
            raise RuntimeError(f'Azimuthal rename references an absent immutable statement: {key!r}')
        if key in replacements:
            raise RuntimeError(f'duplicate Azimuthal rename review: {key!r}')
        original_name = baseline[key].get('name')
        if item.get('baseline_name') != original_name or item.get('current_name') != current_baseline_name(original_name):
            raise RuntimeError(f'Azimuthal rename declaration identity mismatch: {key!r}')
        replacement = str(item['renamed_sha256'])
        if len(replacement) != 64 or any(character not in '0123456789abcdef' for character in replacement):
            raise RuntimeError(f'invalid reviewed Azimuthal AST digest: {key!r}')
        replacements[key] = replacement
    return replacements


def reviewed_local_import_seams(
    module: str,
    module_source: str,
    tree: ast.Module,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Validate and fingerprint every explicitly reviewed function-local import seam."""
    source_lines = module_source.splitlines()
    marker_lines: list[int] = []
    malformed_marker_lines: list[int] = []
    for line_number, line in enumerate(source_lines, start=1):
        if LOCAL_IMPORT_SEAM_MARKER not in line:
            continue
        if line.strip() != LOCAL_IMPORT_SEAM_MARKER:
            malformed_marker_lines.append(line_number)
        else:
            marker_lines.append(line_number)
    if malformed_marker_lines:
        raise RuntimeError(
            f"{module}: local-import seam marker must be the complete comment on lines "
            f"{malformed_marker_lines!r}"
        )

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    imports_by_line: dict[int, list[ast.ImportFrom]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports_by_line.setdefault(node.lineno, []).append(node)

    seams_by_top_level: dict[ast.stmt, list[tuple[int, str, str]]] = {}
    for marker_line in marker_lines:
        import_line = marker_line + 1
        import_nodes = imports_by_line.get(import_line, [])
        if len(import_nodes) != 1:
            raise RuntimeError(
                f"{module}:{marker_line}: local-import seam marker must be immediately "
                "followed by exactly one from-import"
            )
        import_node = import_nodes[0]
        if import_node.level < 1:
            raise RuntimeError(
                f"{module}:{import_line}: reviewed local-import seam must use a relative import"
            )

        marker_indent = source_lines[marker_line - 1][
            : len(source_lines[marker_line - 1])
            - len(source_lines[marker_line - 1].lstrip(" \t"))
        ]
        import_indent = source_lines[import_line - 1][
            : len(source_lines[import_line - 1])
            - len(source_lines[import_line - 1].lstrip(" \t"))
        ]
        if marker_indent != import_indent:
            raise RuntimeError(
                f"{module}:{marker_line}: local-import seam marker and import must have "
                "identical indentation"
            )

        ancestors: list[ast.AST] = []
        cursor: ast.AST = import_node
        while cursor in parents:
            cursor = parents[cursor]
            ancestors.append(cursor)
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ancestors):
            raise RuntimeError(
                f"{module}:{import_line}: reviewed import is not function-local"
            )
        top_level_nodes = [node for node in ancestors if parents.get(node) is tree]
        if len(top_level_nodes) != 1 or not isinstance(
            top_level_nodes[0],
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
        ):
            raise RuntimeError(
                f"{module}:{import_line}: reviewed import must belong to one top-level definition"
            )
        top_level_node = top_level_nodes[0]
        lexical_scope = ".".join(
            node.name
            for node in reversed(ancestors)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        import_ast = stable_ast_dump(import_node)
        seams_by_top_level.setdefault(top_level_node, []).append(
            (import_node.lineno - top_level_node.lineno, lexical_scope, import_ast)
        )

    reviewed: dict[tuple[str, str], tuple[str, str]] = {}
    for top_level_node, seams in seams_by_top_level.items():
        assert isinstance(top_level_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        seam_payload = json.dumps(sorted(seams), separators=(",", ":"))
        reviewed[(module, top_level_node.name)] = (
            digest(top_level_node),
            hashlib.sha256(seam_payload.encode("utf-8")).hexdigest(),
        )
    return reviewed


def reviewed_v21_contract(manifest: dict[str, object]) -> dict[str, object]:
    """Authenticate the additive QSC review without rebaselining history."""
    review = manifest.get('v21_review', {})
    encoded = json.dumps(review, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V21_SHA256:
        raise RuntimeError('v21 review digest mismatch; require explicit review of changed contracts')
    if review.get('release') != '21.0.0':
        raise RuntimeError('v21 review has an unexpected release identity')
    for category, identity in (
        ('definitions', 'name'), ('statements', 'label'),
        ('local_import_seams', 'name'), ('preserved_radial_modules', 'module'),
        ('preserved_radial_definitions', 'qualified_name'),
    ):
        keys = [(item['module'], item[identity]) for item in review[category]]
        if len(keys) != len(set(keys)) or any(not item.get('reason') for item in review[category]):
            raise RuntimeError(f'v21 review has duplicate or unexplained {category}')
    return review


def _reviewed_v21_patch_contract(
    manifest: dict[str, object], v21: dict[str, object], *, key: str,
    release: str, expected_digest: str, previous_digest: str,
    earlier_patches: tuple[dict[str, object], ...] = (),
) -> dict[str, object]:
    """Authenticate patch additions and check every available predecessor pin."""
    review = manifest.get(key, {})
    encoded = json.dumps(review, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != expected_digest:
        raise RuntimeError(f'v{release} review digest mismatch; require explicit review of changed contracts')
    if review.get('release') != release or review.get('previous_review_sha256') != previous_digest:
        raise RuntimeError(f'v{release} review has an unexpected release or predecessor identity')
    historical_definitions = {
        key: record[0] for key, record in REVIEWED_V20_ADDED_DEFINITIONS.items()
    }
    historical_definitions.update({
        (module, name): replacement_hash
        for (module, _baseline), (name, replacement_hash, _reason)
        in REVIEWED_V20_STATEMENT_REPLACEMENTS.items()
    })
    historical_definitions.update({
        key: record[0] for key, record in REVIEWED_LOCAL_IMPORT_SEAMS.items()
    })
    historical_definitions.update({
        (item['module'], item['name']): item['sha256'] for item in v21['definitions']
    })
    historical_statements = {
        key: record[0] for key, record in REVIEWED_V20_ADDED_STATEMENTS.items()
    }
    historical_statements.update({
        (item['module'], item['label']): item['sha256'] for item in v21['statements']
    })
    for earlier in earlier_patches:
        historical_definitions.update({
            (item['module'], item['name']): item['sha256'] for item in earlier['definitions']
        })
        historical_statements.update({
            (item['module'], item['label']): item['sha256'] for item in earlier['statements']
        })
    for category, identity, historical in (
        ('definitions', 'name', historical_definitions),
        ('statements', 'label', historical_statements),
    ):
        keys = [(item['module'], item[identity]) for item in review[category]]
        if len(keys) != len(set(keys)) or any(not item.get('reason') for item in review[category]):
            raise RuntimeError(f'v{release} review has duplicate or unexplained {category}')
        for item in review[category]:
            key = (item['module'], item[identity])
            if key in historical and item.get('previous_sha256') != historical[key]:
                raise RuntimeError(f'v{release} supersession does not match its historical pin: {key[0]}.{key[1]}')
            for field in ('sha256', 'previous_sha256'):
                value = item.get(field)
                if field == 'previous_sha256' and value is None:
                    continue
                if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                    raise RuntimeError(f'v{release} review has an invalid {field}: {key[0]}.{key[1]}')
    return review


def reviewed_v21_patch_contract(manifest: dict[str, object], v21: dict[str, object]) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_1_review', release='21.0.1',
        expected_digest=REVIEWED_V21_0_1_SHA256, previous_digest=REVIEWED_V21_SHA256,
    )


def reviewed_v21_0_2_contract(
    manifest: dict[str, object], v21: dict[str, object], patch: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_2_review', release='21.0.2',
        expected_digest=REVIEWED_V21_0_2_SHA256, previous_digest=REVIEWED_V21_0_1_SHA256,
        earlier_patches=(patch,),
    )


def reviewed_v21_0_3_contract(
    manifest: dict[str, object], v21: dict[str, object],
    first_patch: dict[str, object], second_patch: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_3_review', release='21.0.3',
        expected_digest=REVIEWED_V21_0_3_SHA256, previous_digest=REVIEWED_V21_0_2_SHA256,
        earlier_patches=(first_patch, second_patch),
    )


def reviewed_v21_0_4_contract(
    manifest: dict[str, object], v21: dict[str, object],
    first_patch: dict[str, object], second_patch: dict[str, object],
    third_patch: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_4_review', release='21.0.4',
        expected_digest=REVIEWED_V21_0_4_SHA256, previous_digest=REVIEWED_V21_0_3_SHA256,
        earlier_patches=(first_patch, second_patch, third_patch),
    )


def reviewed_v21_0_5_contract(
    manifest: dict[str, object], v21: dict[str, object],
    first_patch: dict[str, object], second_patch: dict[str, object],
    third_patch: dict[str, object], fourth_patch: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_5_review', release='21.0.5',
        expected_digest=REVIEWED_V21_0_5_SHA256, previous_digest=REVIEWED_V21_0_4_SHA256,
        earlier_patches=(first_patch, second_patch, third_patch, fourth_patch),
    )


def reviewed_v21_0_6_contract(
    manifest: dict[str, object], v21: dict[str, object],
    first_patch: dict[str, object], second_patch: dict[str, object],
    third_patch: dict[str, object], fourth_patch: dict[str, object],
    fifth_patch: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_0_6_review', release='21.0.6',
        expected_digest=REVIEWED_V21_0_6_SHA256, previous_digest=REVIEWED_V21_0_5_SHA256,
        earlier_patches=(first_patch, second_patch, third_patch, fourth_patch, fifth_patch),
    )


def reviewed_v21_1_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_1_review', release='21.1.0',
        expected_digest=REVIEWED_V21_1_SHA256, previous_digest=REVIEWED_V21_0_6_SHA256,
        earlier_patches=earlier_patches,
    )


def reviewed_v21_1_1_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_1_1_review', release='21.1.1',
        expected_digest=REVIEWED_V21_1_1_SHA256, previous_digest=REVIEWED_V21_1_SHA256,
        earlier_patches=earlier_patches,
    )


def reviewed_v21_1_2_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    return _reviewed_v21_patch_contract(
        manifest, v21, key='v21_1_2_review', release='21.1.2',
        expected_digest=REVIEWED_V21_1_2_SHA256, previous_digest=REVIEWED_V21_1_1_SHA256,
        earlier_patches=earlier_patches,
    )

def reviewed_v22_augmentation_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    review = _reviewed_v21_patch_contract(
        manifest, v21, key='v22_augmentation_review', release='22.0.0',
        expected_digest=REVIEWED_V22_AUGMENTATION_SHA256,
        previous_digest=REVIEWED_V21_1_1_SHA256,
        earlier_patches=earlier_patches,
    )
    relocations = review.get('definition_relocations', ())
    keys = [(item['module'], item['name']) for item in relocations]
    if len(keys) != len(set(keys)) or set(keys) != set(REVIEWED_PRESERVED_AUGMENTATION_DEFINITIONS):
        raise RuntimeError('v22 augmentation relocation coverage differs from the preserved PTA definitions')
    destinations = {(item['module'], item['name']): item for item in review['definitions']}
    for item in relocations:
        key = (item['module'], item['name'])
        if item.get('previous_sha256') != REVIEWED_PRESERVED_AUGMENTATION_DEFINITIONS[key]:
            raise RuntimeError(f'v22 augmentation relocation does not match its preserved predecessor: {key[0]}.{key[1]}')
        destination = destinations.get((item.get('destination_module'), item.get('destination_name')))
        if not item.get('reason') or destination is None or destination['sha256'] != item.get('sha256'):
            raise RuntimeError(f'v22 augmentation relocation has no matching reviewed destination: {key[0]}.{key[1]}')
        if item['module'] == item['destination_module'] or item['name'] != item['destination_name']:
            raise RuntimeError(f'v22 augmentation relocation has an unexpected owner or public name: {key[0]}.{key[1]}')
    return review


def reviewed_v22_coverage_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    """Authenticate coverage planning while preserving the complete prior record."""
    keys = (*REVIEWED_COMMON_INVENTORY_KEYS, 'v22_augmentation_review')
    prior = {key: manifest[key] for key in keys if key in manifest}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_COVERAGE_PREDECESSOR_SHA256:
        raise RuntimeError('v22 coverage predecessor inventory changed; preserve every historical record')
    review = _reviewed_v21_patch_contract(
        manifest, v21, key='v22_coverage_review', release='22.0.0',
        expected_digest=REVIEWED_V22_COVERAGE_SHA256,
        previous_digest=REVIEWED_V22_AUGMENTATION_SHA256,
        earlier_patches=earlier_patches,
    )
    if (review.get('predecessor_commit') != REVIEWED_V22_COVERAGE_PREDECESSOR_COMMIT
            or review.get('feature') != 'coverage-sampling'):
        raise RuntimeError('v22 coverage review has an unexpected development predecessor or feature')
    validation_tools = review.get('validation_tools', ())
    paths = [item.get('path') for item in validation_tools]
    if paths != ['tools/certify_qsc_lipschitz.py'] or any(not item.get('reason') for item in validation_tools):
        raise RuntimeError('v22 coverage validation-tool review has missing, duplicate or unexplained paths')
    for item in validation_tools:
        value = item.get('sha256')
        if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
            raise RuntimeError('v22 coverage validation-tool review has an invalid digest')
    return review


def verify_coverage_validation_tools(review) -> None:
    """The full-domain rational certificate is independently reviewed source."""
    for item in review['validation_tools']:
        path = ROOT / item['path']
        if not path.is_file() or hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest() != item['sha256']:
            raise RuntimeError(f'v22 coverage validation tool changed or is missing: {item["path"]}')


def reviewed_v22_release_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    """Join independently authenticated parents without rewriting either chain."""
    manifest = _without_reviewed_pta_successor(manifest)
    expected_keys = {'v22_release_review'}
    for parent in REVIEWED_V22_RELEASE_PARENTS:
        keys = parent['inventory_keys']
        expected_keys.update(keys)
        snapshot = {key: manifest[key] for key in keys if key in manifest}
        encoded = json.dumps(snapshot, sort_keys=True, separators=(',', ':')).encode('utf-8')
        if hashlib.sha256(encoded).hexdigest() != parent['inventory_sha256']:
            raise RuntimeError(f'v22 release parent inventory changed: {parent["commit"]}')
    successor_key = 'v22_policy_memory_review'
    throughput_key = 'v22_policy_throughput_review'
    radial_key = 'v22_radial_retirement_review'
    window_key = 'v22_policy_window_review'
    tilted_key = 'v22_tilted_azimuthal_gpu_review'
    if set(manifest) - {successor_key, throughput_key, radial_key, window_key, tilted_key} != expected_keys:
        raise RuntimeError('v22 release inventory keyset differs from its reviewed parent union')
    review = _reviewed_v21_patch_contract(
        manifest, v21, key='v22_release_review', release='22.0.0',
        expected_digest=REVIEWED_V22_RELEASE_SHA256,
        previous_digest=REVIEWED_V22_COVERAGE_SHA256,
        earlier_patches=earlier_patches,
    )
    if review.get('parent_snapshots') != list(REVIEWED_V22_RELEASE_PARENTS):
        raise RuntimeError('v22 release has unexpected parent snapshots')
    if review.get('merge_resolution') != {
        'shared_predecessor': 'v21_1_1_review',
        'main_reviews': ['v21_1_2_review'],
        'feature_reviews': ['v22_augmentation_review', 'v22_coverage_review'],
        'runtime_resolution': 'Preserve every independently pinned runtime definition and binding from both parents; supersede only the five release identity bindings.',
    }:
        raise RuntimeError('v22 release has an unexpected merge resolution')
    bindings = {
        ('__init__', '__version__'), ('cli', 'SCRIPT_VERSION'),
        ('cli', 'SCRIPT_BASENAME'), ('config', 'SCRIPT_VERSION'),
        ('config', 'SCRIPT_VERSION_COMPACT'),
    }
    if (review['definitions'] or review.get('complete_modules')
            or {(item['module'], item.get('binding')) for item in review['statements']} != bindings
            or len(review['statements']) != len(bindings)):
        raise RuntimeError('v22 release must review exactly its five release identity bindings')
    if successor_key in manifest:
        # The historical release permits only an independently authenticated
        # successor. An arbitrary new appendix must not weaken its exact keyset.
        reviewed_v22_policy_memory_contract(manifest, v21, *earlier_patches, review)
    elif throughput_key in manifest or radial_key in manifest or window_key in manifest or tilted_key in manifest:
        raise RuntimeError('v22 policy throughput successor requires the reviewed memory predecessor')
    return review


def reviewed_v22_policy_memory_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    """Authenticate bounded policy parents without rewriting the release merge."""
    manifest = _without_reviewed_pta_successor(manifest)
    key = 'v22_policy_memory_review'
    successor_key = 'v22_policy_throughput_review'
    radial_key = 'v22_radial_retirement_review'
    window_key = 'v22_policy_window_review'
    tilted_key = 'v22_tilted_azimuthal_gpu_review'
    prior = {name: value for name, value in manifest.items() if name not in (key, successor_key, radial_key, window_key, tilted_key)}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_POLICY_MEMORY_PREDECESSOR_SHA256:
        raise RuntimeError('v22 policy memory predecessor inventory changed; preserve every historical record')
    review = _reviewed_v21_patch_contract(
        manifest, v21, key=key, release='22.0.0',
        expected_digest=REVIEWED_V22_POLICY_MEMORY_SHA256,
        previous_digest=REVIEWED_V22_RELEASE_SHA256,
        earlier_patches=earlier_patches,
    )
    if (review.get('predecessor_commit') != REVIEWED_V22_POLICY_MEMORY_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_POLICY_MEMORY_PREDECESSOR_SHA256
            or review.get('feature') != 'bounded-policy-parent-memory'):
        raise RuntimeError('v22 policy memory review has an unexpected predecessor or feature')
    expected_definitions = {
        ('pipeline', '_main_impl'),
        ('publication_memory', 'publication_ram_headroom'),
        ('publication_memory', 'native_fullframe_dense_reserve'),
        ('publication_memory', 'policy_parent_memory_plan'),
        ('tta_scheduler', 'TtaSchedulerState'), ('tta_scheduler', 'TtaScheduler'),
    }
    expected_statements = {
        ('pipeline', 'import_dataclasses_replace'),
        ('pipeline', 'import_.publication_memory'),
    }
    if ({(item['module'], item['name']) for item in review['definitions']} != expected_definitions
            or {(item['module'], item['label']) for item in review['statements']} != expected_statements
            or any(review.get(category) for category in (
                'local_import_seam_updates', 'preserved_radial_definition_updates',
                'preserved_radial_module_updates', 'complete_modules'))):
        raise RuntimeError('v22 policy memory review must cover exactly its parent-admission definitions and imports')
    if successor_key in manifest:
        reviewed_v22_policy_throughput_contract(manifest, v21, *earlier_patches, review)
    elif radial_key in manifest or window_key in manifest or tilted_key in manifest:
        raise RuntimeError('v22 Radial retirement successor requires the reviewed throughput predecessor')
    return review


def reviewed_v22_policy_throughput_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    """Authenticate the shared-policy correction against its distributed bundle."""
    manifest = _without_reviewed_pta_successor(manifest)
    key = 'v22_policy_throughput_review'
    successor_key = 'v22_radial_retirement_review'
    window_key = 'v22_policy_window_review'
    tilted_key = 'v22_tilted_azimuthal_gpu_review'
    prior = {name: value for name, value in manifest.items() if name not in (key, successor_key, window_key, tilted_key)}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_POLICY_THROUGHPUT_PREDECESSOR_SHA256:
        raise RuntimeError('v22 policy throughput predecessor inventory changed; preserve every historical record')
    review = _reviewed_v21_patch_contract(
        manifest, v21, key=key, release='22.0.0',
        expected_digest=REVIEWED_V22_POLICY_THROUGHPUT_SHA256,
        previous_digest=REVIEWED_V22_POLICY_MEMORY_SHA256,
        earlier_patches=earlier_patches,
    )
    if (review.get('predecessor_bundle') != {
                'name': 'XTA_v22.0.0_complete_source.zip',
                'sha256': REVIEWED_V22_POLICY_THROUGHPUT_PREDECESSOR_BUNDLE_SHA256,
            }
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_POLICY_THROUGHPUT_PREDECESSOR_SHA256
            or 'predecessor_commit' in review
            or review.get('feature') != 'shared-policy-union-throughput'):
        raise RuntimeError('v22 policy throughput review has an unexpected bundle predecessor or feature')
    expected_definitions = {
        ('pipeline', '_main_impl'),
        ('inference', '_DeviceUnionAccumulator'),
        ('inference', '_direct_predict_stream'),
        ('inference', 'predict_source_and_accumulate'),
        ('runtime', '_attach_memfd_transfers_to_task'),
        ('runtime', '_materialize_worker_task_memfd_paths'),
        ('tta_augmentation_runtime', '_CoverageWriter'),
        ('tta_augmentation_runtime', '_validate_policy_parent_group'),
        ('tta_augmentation_runtime', '_open_policy_sibling_outputs'),
        ('tta_augmentation_runtime', 'predict_policy_source'),
        ('tta_scheduler', 'TtaScheduler'),
        ('publication_memory', 'native_fullframe_dense_reserve'),
        ('publication_memory', 'policy_parent_memory_plan'),
    }
    if ({(item['module'], item['name']) for item in review['definitions']} != expected_definitions
            or {(item['module'], item['label']) for item in review['statements']}
                != {('pipeline', 'import_.inference')}
            or {(item['module'], item['name']) for item in review.get('local_import_seam_updates', ())}
                != {('inference', 'predict_source_and_accumulate')}
            or any(review.get(category) for category in (
                'preserved_radial_definition_updates',
                'preserved_radial_module_updates', 'complete_modules'))):
        raise RuntimeError('v22 policy throughput review must cover exactly its shared-policy definitions, import and seam')
    if successor_key in manifest:
        reviewed_v22_radial_retirement_contract(manifest, v21, *earlier_patches, review)
    elif window_key in manifest or tilted_key in manifest:
        raise RuntimeError('v22 policy window successor requires the reviewed Radial retirement predecessor')
    return review


def reviewed_v22_radial_retirement_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    """Authenticate Radial GPU retirement without altering its numerical history."""
    manifest = _without_reviewed_pta_successor(manifest)
    key = 'v22_radial_retirement_review'
    successor_key = 'v22_policy_window_review'
    tilted_key = 'v22_tilted_azimuthal_gpu_review'
    prior = {name: value for name, value in manifest.items() if name not in (key, successor_key, tilted_key)}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_RADIAL_RETIREMENT_PREDECESSOR_SHA256:
        raise RuntimeError('v22 Radial retirement predecessor inventory changed; preserve every historical record')
    review = _reviewed_v21_patch_contract(
        manifest, v21, key=key, release='22.0.0',
        expected_digest=REVIEWED_V22_RADIAL_RETIREMENT_SHA256,
        previous_digest=REVIEWED_V22_POLICY_THROUGHPUT_SHA256,
        earlier_patches=earlier_patches,
    )
    if (review.get('predecessor_bundle') != {
                'name': 'XTA_v22.0.0_complete_source.zip',
                'sha256': REVIEWED_V22_RADIAL_RETIREMENT_PREDECESSOR_BUNDLE_SHA256,
            }
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_RADIAL_RETIREMENT_PREDECESSOR_SHA256
            or 'predecessor_commit' in review
            or review.get('feature') != 'radial-gpu-retirement'):
        raise RuntimeError('v22 Radial retirement review has an unexpected bundle predecessor or feature')
    expected_definitions = {
        ('backprojection', '_MainProcessGpuStageCoordinator'),
        ('cylindrical_projection', '_ordered_radial_blocks'),
        ('cylindrical_projection', '_try_radial_cuda_stage'),
        ('cylindrical_projection', '_ordered_radial_cuda_blocks'),
        ('cylindrical_projection', 'backproject_radial_volume_to_volume'),
    }
    expected_statements = {
        ('cylindrical_projection', 'shared_future_import'),
        ('cylindrical_projection', 'binding__CUDA_RECHECK_SLICES'),
        ('cylindrical_projection', 'binding__CUDA_RECHECK_SECONDS'),
    }
    if ({(item['module'], item['name']) for item in review['definitions']} != expected_definitions
            or {(item['module'], item['label']) for item in review['statements']} != expected_statements
            or [item['module'] for item in review.get('preserved_radial_module_updates', ())]
                != ['cylindrical_projection']
            or any(review.get(category) for category in (
                'local_import_seam_updates', 'preserved_radial_definition_updates', 'complete_modules'))):
        raise RuntimeError('v22 Radial retirement review must cover exactly its scheduler, projection and retry contracts')
    if successor_key in manifest:
        reviewed_v22_policy_window_contract(manifest, v21, *earlier_patches, review)
    elif tilted_key in manifest:
        raise RuntimeError('v22 Tilted Azimuthal GPU successor requires the reviewed policy window predecessor')
    return review


def reviewed_v22_policy_window_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    """Authenticate the policy-only 384 GiB request against its released bundle."""
    manifest = _without_reviewed_pta_successor(manifest)
    key = 'v22_policy_window_review'
    successor_key = 'v22_tilted_azimuthal_gpu_review'
    prior = {name: value for name, value in manifest.items() if name not in (key, successor_key)}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_POLICY_WINDOW_PREDECESSOR_SHA256:
        raise RuntimeError('v22 policy window predecessor inventory changed; preserve every historical record')
    review = _reviewed_v21_patch_contract(
        manifest, v21, key=key, release='22.0.0',
        expected_digest=REVIEWED_V22_POLICY_WINDOW_SHA256,
        previous_digest=REVIEWED_V22_RADIAL_RETIREMENT_SHA256,
        earlier_patches=earlier_patches,
    )
    if (review.get('predecessor_bundle') != {
                'name': 'XTA_v22.0.0_complete_source.zip',
                'sha256': REVIEWED_V22_POLICY_WINDOW_PREDECESSOR_BUNDLE_SHA256,
            }
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_POLICY_WINDOW_PREDECESSOR_SHA256
            or review.get('preserved_pipeline_statements_sha256') != REVIEWED_V22_POLICY_WINDOW_PRESERVED_PIPELINE_SHA256
            or 'predecessor_commit' in review
            or review.get('feature') != 'policy-parent-window-384-gib'):
        raise RuntimeError('v22 policy window review has an unexpected bundle predecessor or feature')
    if ([(item['module'], item['name']) for item in review['definitions']]
            != [('pipeline', '_main_impl')]
            or any(review.get(category) for category in (
                'statements', 'local_import_seam_updates', 'preserved_radial_definition_updates',
                'preserved_radial_module_updates', 'complete_modules'))):
        raise RuntimeError('v22 policy window review must cover only pipeline._main_impl and no imports or other runtime changes')
    if successor_key in manifest:
        reviewed_v22_tilted_azimuthal_gpu_contract(manifest, v21, *earlier_patches, review)
    return review


def reviewed_v22_tilted_azimuthal_gpu_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    """Authenticate the bounded GPU operator against the complete prior commit."""
    manifest = _without_reviewed_pta_successor(manifest)
    key = 'v22_tilted_azimuthal_gpu_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_TILTED_AZIMUTHAL_PREDECESSOR_SHA256:
        raise RuntimeError('v22 Tilted Azimuthal GPU predecessor inventory changed; preserve every historical record')
    review = _reviewed_v21_patch_contract(
        manifest, v21, key=key, release='22.0.0',
        expected_digest=REVIEWED_V22_TILTED_AZIMUTHAL_SHA256,
        previous_digest=REVIEWED_V22_POLICY_WINDOW_SHA256,
        earlier_patches=earlier_patches,
    )
    if (review.get('predecessor_commit') != REVIEWED_V22_TILTED_AZIMUTHAL_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_TILTED_AZIMUTHAL_PREDECESSOR_SHA256
            or review.get('feature') != 'tilted-azimuthal-gpu-projection'):
        raise RuntimeError('v22 Tilted Azimuthal GPU review has an unexpected predecessor or feature')
    modules = {'tilted_azimuthal_projection', 'tilted_azimuthal_projection_cuda'}
    definitions = {
        ('backprojection', '_MainProcessGpuStageCoordinator'),
        ('backprojection', '_TiltedAzimuthalCudaStage'),
        ('backprojection', '_try_tilted_azimuthal_cuda_stage'),
        ('backprojection', '_ordered_tilted_azimuthal_coordinates'),
        ('backprojection', '_project_tilted_azimuthal_sink'),
        ('backprojection', '_backproject_tilted_azimuthal_volume_to_volume'),
        ('pipeline', '_execution_runtime_provenance'),
        ('tilted_azimuthal_projection', 'TiltedAzimuthalPlanUnavailable'),
        ('tilted_azimuthal_projection', 'TiltedAzimuthalProjectionPlan'),
        ('tilted_azimuthal_projection', 'build_tilted_azimuthal_plan'),
        ('tilted_azimuthal_projection_cuda', 'TiltedAzimuthalCudaProjectionUnavailable'),
        ('tilted_azimuthal_projection_cuda', 'TiltedAzimuthalCudaProjectionUnsafeFailure'),
        ('tilted_azimuthal_projection_cuda', '_TiltedAzimuthalContract'),
        ('tilted_azimuthal_projection_cuda', '_shape'),
        ('tilted_azimuthal_projection_cuda', '_validate_tilted_azimuthal_contract'),
        ('tilted_azimuthal_projection_cuda', '_validate_initial_packed'),
        ('tilted_azimuthal_projection_cuda', '_frame_row_bands'),
        ('tilted_azimuthal_projection_cuda', '_preflight_case'),
        ('tilted_azimuthal_projection_cuda', 'TiltedAzimuthalCudaProjector'),
    }
    existing_statements = {
        ('backprojection', 'import_concurrent_futures'),
        ('backprojection', 'binding__TILTED_AZIMUTHAL_CUDA_RECHECK_FRAMES'),
        ('backprojection', 'binding__TILTED_AZIMUTHAL_CUDA_RECHECK_SECONDS'),
    }
    if ({(item['module'], item['name']) for item in review['definitions']} != definitions
            or {(item['module'], item['label']) for item in review['statements'] if item['module'] not in modules}
                != existing_statements
            or sorted(review.get('complete_modules', ())) != sorted(modules)
            or any(review.get(category) for category in (
                'local_import_seam_updates', 'preserved_radial_definition_updates', 'preserved_radial_module_updates'))):
        raise RuntimeError('v22 Tilted Azimuthal GPU review must cover exactly its operator, routing and provenance contracts')
    return review


def reviewed_v22_pta_throughput_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    """Authenticate the PTA successor and its complete distributed predecessor."""
    manifest = _without_reviewed_v22_1_release(manifest)
    key = 'v22_pta_throughput_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_PTA_THROUGHPUT_PREDECESSOR_SHA256:
        raise RuntimeError('v22 PTA throughput predecessor inventory changed; preserve every historical record')
    # Historical entry points may authenticate this successor before they have
    # traversed the complete review chain. The whole prior snapshot above is
    # pinned, so reconstruct its effective definitions in the reviewed order.
    ordered_keys = (
        *(f'v21_0_{index}_review' for index in range(1, 7)),
        'v21_1_review', 'v21_1_1_review', 'v21_1_2_review',
        'v22_augmentation_review', 'v22_coverage_review', 'v22_release_review',
        'v22_policy_memory_review', 'v22_policy_throughput_review',
        'v22_radial_retirement_review', 'v22_policy_window_review',
        'v22_tilted_azimuthal_gpu_review',
    )
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='22.0.0',
        expected_digest=REVIEWED_V22_PTA_THROUGHPUT_SHA256,
        previous_digest=REVIEWED_V22_TILTED_AZIMUTHAL_SHA256,
        earlier_patches=tuple(prior[name] for name in ordered_keys),
    )
    if (review.get('predecessor_bundle') != {
                'name': 'XTA_v22.0.0_complete_source.zip',
                'sha256': REVIEWED_V22_PTA_THROUGHPUT_PREDECESSOR_BUNDLE_SHA256,
            }
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_PTA_THROUGHPUT_PREDECESSOR_SHA256
            or 'predecessor_commit' in review
            or review.get('feature') != 'pta-pipelined-gpu-publication-and-output-formats'):
        raise RuntimeError('v22 PTA throughput review has an unexpected bundle predecessor or feature')
    modules = {'pta_batch_pipeline', 'pta_gpu_publication'}
    if ({(item['module'], item['name']) for item in review['definitions']} != REVIEWED_V22_PTA_DEFINITION_KEYS
            or {(item['module'], item['label']) for item in review['statements']} != REVIEWED_V22_PTA_STATEMENT_KEYS
            or sorted(review.get('complete_modules', ())) != sorted(modules)
            or any(review.get(category) for category in (
                'local_import_seam_updates', 'preserved_radial_definition_updates',
                'preserved_radial_module_updates'))):
        raise RuntimeError('v22 PTA throughput review must cover exactly its publication, CPU budget and format contracts')
    snapshots = review.get('module_snapshots', ())
    identities = [item.get('module') for item in snapshots]
    if (len(identities) != len(set(identities))
            or set(identities) != set(REVIEWED_V22_PTA_PREDECESSOR_MODULES)):
        raise RuntimeError('v22 PTA throughput source snapshot coverage differs')
    for item in snapshots:
        if item.get('previous_ast_sha256') != REVIEWED_V22_PTA_PREDECESSOR_MODULES[item['module']]:
            raise RuntimeError(f"v22 PTA throughput source predecessor changed: {item['module']}")
        value = item.get('ast_sha256')
        if (not item.get('reason') or not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)):
            raise RuntimeError('v22 PTA throughput source snapshot has no reason or valid digest')
    return review


def _authenticate_successor_once(manifest, key, contract):
    """Reuse a successful appendix check only for the same pass and full context.

    Review contracts read their input and the historical wrappers make shallow
    copies. The ordered identities cover every predecessor value, so a wrapper
    presented with a different predecessor context must authenticate again.
    Direct contract calls outside ``main`` are never cached.
    """
    cache = _ACTIVE_REVIEW_AUTH_CACHE.get()
    if cache is None:
        return contract(manifest, manifest['v21_review'])
    context = (key, tuple((name, id(value)) for name, value in manifest.items()))
    if context in cache:
        return manifest[key]
    result = contract(manifest, manifest['v21_review'])
    cache.add(context)
    return result


def _without_reviewed_pta_successor(manifest: dict[str, object]) -> dict[str, object]:
    """Historical contracts permit only the fully authenticated new appendix."""
    manifest = _without_reviewed_v22_1_release(manifest)
    key = 'v22_pta_throughput_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v22_pta_throughput_contract)
    return {name: value for name, value in manifest.items() if name != key}


def reviewed_v22_1_release_contract(
    manifest: dict[str, object], v21: dict[str, object],
    *earlier_patches: dict[str, object],
) -> dict[str, object]:
    """Promote the exact distributed PTA candidate through five version bindings."""
    manifest = _without_reviewed_v22_2_release(manifest)
    key = 'v22_1_release_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_1_RELEASE_PREDECESSOR_SHA256:
        raise RuntimeError('v22.1.0 release predecessor inventory changed; preserve every historical record')
    keys = (
        *(f'v21_0_{index}_review' for index in range(1, 7)),
        'v21_1_review', 'v21_1_1_review', 'v21_1_2_review',
        'v22_augmentation_review', 'v22_coverage_review', 'v22_release_review',
        'v22_policy_memory_review', 'v22_policy_throughput_review',
        'v22_radial_retirement_review', 'v22_policy_window_review',
        'v22_tilted_azimuthal_gpu_review', 'v22_pta_throughput_review',
    )
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='22.1.0',
        expected_digest=REVIEWED_V22_1_RELEASE_SHA256,
        previous_digest=REVIEWED_V22_PTA_THROUGHPUT_SHA256,
        earlier_patches=tuple(prior[name] for name in keys),
    )
    if (review.get('predecessor_bundle') != {
                'name': 'XTA_v22.0.0_complete_source.zip',
                'sha256': REVIEWED_V22_1_RELEASE_PREDECESSOR_BUNDLE_SHA256,
            }
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_1_RELEASE_PREDECESSOR_SHA256
            or review.get('feature') != 'release-22.1.0-consolidated-pta-backends'
            or review.get('preserved_module_statements_sha256') != REVIEWED_V22_1_RELEASE_PRESERVED_MODULES
            or 'predecessor_commit' in review):
        raise RuntimeError('v22.1.0 release has an unexpected bundle predecessor, source scope or feature')
    bindings = {('__init__', '__version__'), ('cli', 'SCRIPT_VERSION'),
                ('cli', 'SCRIPT_BASENAME'), ('config', 'SCRIPT_VERSION'),
                ('config', 'SCRIPT_VERSION_COMPACT')}
    if (review['definitions'] or len(review['statements']) != len(bindings)
            or {(item['module'], item.get('binding')) for item in review['statements']} != bindings
            or any(review.get(category) for category in (
                'local_import_seam_updates', 'preserved_radial_definition_updates',
                'preserved_radial_module_updates', 'complete_modules', 'module_snapshots'))):
        raise RuntimeError('v22.1.0 release must review exactly its five release identity bindings')
    return review


def _without_reviewed_v22_1_release(manifest: dict[str, object]) -> dict[str, object]:
    """Admit the version-only successor without weakening historical snapshots."""
    manifest = _without_reviewed_v22_2_release(manifest)
    key = 'v22_1_release_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v22_1_release_contract)
    return {name: value for name, value in manifest.items() if name != key}


def reviewed_v24_release_contract(manifest, v21, *earlier_patches):
    """Authenticate the semantic release against the complete v22.3.2 inventory."""
    manifest = _without_reviewed_v24_0_2_release(manifest)
    key = 'v24_release_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V24_RELEASE_PREDECESSOR_SHA256:
        raise RuntimeError('v24.0.1 predecessor inventory changed; preserve every historical record')
    ordered_keys = (
        *(f'v21_0_{index}_review' for index in range(1, 7)),
        'v21_1_review', 'v21_1_1_review', 'v21_1_2_review',
        'v22_augmentation_review', 'v22_coverage_review', 'v22_release_review',
        'v22_policy_memory_review', 'v22_policy_throughput_review',
        'v22_radial_retirement_review', 'v22_policy_window_review',
        'v22_tilted_azimuthal_gpu_review', 'v22_pta_throughput_review',
        'v22_1_release_review', 'v22_2_release_review', 'v22_3_release_review',
        'v22_3_1_release_review', 'v22_3_2_release_review',
    )
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='24.0.1',
        expected_digest=REVIEWED_V24_RELEASE_SHA256,
        previous_digest=REVIEWED_V22_3_2_RELEASE_SHA256,
        earlier_patches=tuple(prior[name] for name in ordered_keys),
    )
    if (review.get('predecessor_commit') != REVIEWED_V24_RELEASE_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V24_RELEASE_PREDECESSOR_SHA256
            or review.get('feature') != 'yolo-semantic-segmentation'):
        raise RuntimeError('v24.0.1 review has an unexpected predecessor or feature')
    snapshots = review.get('module_snapshots', ())
    modules = [item.get('module') for item in snapshots]
    if len(modules) != len(set(modules)) or set(modules) != set(REVIEWED_V24_RELEASE_PREDECESSOR_MODULES):
        raise RuntimeError('v24.0.1 source snapshot coverage differs')
    for item in snapshots:
        module = item['module']
        previous = REVIEWED_V24_RELEASE_PREDECESSOR_MODULES[module]
        historical = item.get('previous_top_level', ())
        historical_digest = hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest()
        if (item.get('previous_ast_sha256') != previous['ast_sha256']
                or historical_digest != previous['statements_sha256']):
            raise RuntimeError(f'v24.0.1 source predecessor changed: {module}')
        if 'removed' in item:
            raise RuntimeError(f'v24.0.1 has an unreviewed module removal: {module}')
        for value in (item.get('ast_sha256'), *historical, *item.get('top_level', ())):
            if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise RuntimeError(f'v24.0.1 source snapshot has an invalid digest: {module}')
        if not item.get('reason'):
            raise RuntimeError(f'v24.0.1 source snapshot has no review reason: {module}')
        positions = []
        for record in review['definitions'] + review['statements']:
            if record['module'] != module:
                continue
            current_index = record.get('current_index')
            previous_index = record.get('previous_index')
            if (type(current_index) is not int or not 0 <= current_index < len(item['top_level'])
                    or item['top_level'][current_index] != record['sha256']):
                raise RuntimeError(f'v24.0.1 statement position differs: {module}')
            if previous_index is None:
                if record['previous_sha256'] is not None:
                    raise RuntimeError(f'v24.0.1 new statement has an unexpected predecessor: {module}')
            elif (type(previous_index) is not int or not 0 <= previous_index < len(historical)
                    or historical[previous_index] != record['previous_sha256']):
                raise RuntimeError(f'v24.0.1 statement predecessor changed: {module}')
            positions.append(current_index)
        if len(positions) != len(set(positions)):
            raise RuntimeError(f'v24.0.1 source has duplicate reviewed positions: {module}')
    if any(record['module'] not in set(modules) for record in review['definitions'] + review['statements']):
        raise RuntimeError('v24.0.1 statement has no complete source snapshot')
    if review.get('removed_definitions', ()):
        raise RuntimeError('v24.0.1 cannot retire a definition outside its tagged review')
    reviewed_radial_module_hashes(prior['v21_review'], (*tuple(prior[name] for name in ordered_keys), review))
    reviewed_radial_definition_hashes(prior['v21_review'], (*tuple(prior[name] for name in ordered_keys), review))
    validation_tools = review.get('validation_tools', ())
    prior_tools = {item['path']: item['sha256'] for item in prior['v22_3_2_release_review']['validation_tools']}
    if [item.get('path') for item in validation_tools] != [
            *prior_tools, 'tools/export_semantic_logits.py', 'tools/qualify_semantic_trt.py',
            'tools/qualify_pta_classification.py', 'tools/qualify_pta_gpu_masks.py',
            'tools/qualify_pta_gpu_render.py']:
        raise RuntimeError('v24.0.1 validation-tool review has missing or duplicate paths')
    for item in validation_tools:
        if item.get('previous_sha256') != prior_tools.get(item['path']):
            raise RuntimeError('v24.0.1 validation-tool predecessor changed')
        value = item.get('sha256')
        if (not item.get('reason') or not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)):
            raise RuntimeError('v24.0.1 validation-tool review has an invalid digest or reason')
    return review


def reviewed_v24_0_2_release_contract(manifest, v21, *earlier_patches):
    """Authenticate the patch as a successor to the tagged 24.0.1 receipt."""
    manifest = _without_reviewed_v24_0_3_release(manifest)
    key = 'v24_0_2_release_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V24_0_2_RELEASE_PREDECESSOR_SHA256:
        raise RuntimeError('v24.0.2 predecessor inventory changed; preserve every historical record')
    reviewed_v24_release_contract(prior, prior['v21_review'])
    earlier = tuple(value for name, value in prior.items()
                    if name != 'v21_review' and isinstance(value, dict)
                    and 'release' in value and 'definitions' in value)
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='24.0.2',
        expected_digest=REVIEWED_V24_0_2_RELEASE_SHA256,
        previous_digest=REVIEWED_V24_RELEASE_SHA256, earlier_patches=earlier)
    if (review.get('predecessor_commit') != REVIEWED_V24_0_2_RELEASE_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V24_0_2_RELEASE_PREDECESSOR_SHA256
            or review.get('feature') != 'repository-review-corrections'):
        raise RuntimeError('v24.0.2 review has an unexpected predecessor or feature')
    snapshots = review.get('module_snapshots', ())
    modules = [item.get('module') for item in snapshots]
    if len(modules) != len(set(modules)) or set(modules) != set(REVIEWED_V24_0_2_RELEASE_PREDECESSOR_MODULES):
        raise RuntimeError('v24.0.2 source snapshot coverage differs')
    for item in snapshots:
        module = item['module']
        previous = REVIEWED_V24_0_2_RELEASE_PREDECESSOR_MODULES[module]
        historical = item.get('previous_top_level', ())
        historical_digest = hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest()
        if (item.get('previous_ast_sha256') != previous['ast_sha256']
                or historical_digest != previous['statements_sha256']):
            raise RuntimeError(f'v24.0.2 source predecessor changed: {module}')
        if item.get('removed'):
            raise RuntimeError(f'v24.0.2 has an unreviewed module removal: {module}')
        for value in (item.get('ast_sha256'), *historical, *item.get('top_level', ())):
            if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise RuntimeError(f'v24.0.2 source snapshot has an invalid digest: {module}')
        if not item.get('reason'):
            raise RuntimeError(f'v24.0.2 source snapshot has no review reason: {module}')
        positions = []
        for record in review['definitions'] + review['statements']:
            if record['module'] != module:
                continue
            current_index, previous_index = record.get('current_index'), record.get('previous_index')
            if (type(current_index) is not int or not 0 <= current_index < len(item['top_level'])
                    or item['top_level'][current_index] != record['sha256']):
                raise RuntimeError(f'v24.0.2 statement position differs: {module}')
            if previous_index is None:
                if record['previous_sha256'] is not None:
                    raise RuntimeError(f'v24.0.2 new statement has an unexpected predecessor: {module}')
            elif (type(previous_index) is not int or not 0 <= previous_index < len(historical)
                    or historical[previous_index] != record['previous_sha256']):
                raise RuntimeError(f'v24.0.2 statement predecessor changed: {module}')
            positions.append(current_index)
        if len(positions) != len(set(positions)):
            raise RuntimeError(f'v24.0.2 source has duplicate reviewed positions: {module}')
    if any(record['module'] not in set(modules) for record in review['definitions'] + review['statements']):
        raise RuntimeError('v24.0.2 statement has no complete source snapshot')
    retired = review.get('removed_definitions', ())
    if [(item.get('module'), item.get('name')) for item in retired] != [('geometry', '_angle_from_aug_id')]:
        raise RuntimeError('v24.0.2 reviewed definition removals differ')
    for item in retired:
        snapshot = next(value for value in snapshots if value['module'] == item['module'])
        index = item.get('previous_index')
        if (type(index) is not int or not 0 <= index < len(snapshot['previous_top_level'])
                or item.get('previous_sha256') != snapshot['previous_top_level'][index]
                or not item.get('reason')):
            raise RuntimeError('v24.0.2 retired definition predecessor changed')
    removed_statements = review.get('removed_statements', ())
    if len({(item.get('module'), item.get('previous_index')) for item in removed_statements}) != len(removed_statements):
        raise RuntimeError('v24.0.2 has duplicate retired statements')
    for item in removed_statements:
        snapshot = next((value for value in snapshots if value['module'] == item.get('module')), None)
        index = item.get('previous_index')
        if (snapshot is None or type(index) is not int
                or not 0 <= index < len(snapshot['previous_top_level'])
                or item.get('previous_sha256') != snapshot['previous_top_level'][index]
                or not item.get('reason')):
            raise RuntimeError('v24.0.2 retired statement predecessor changed')
    reviewed_radial_module_hashes(prior['v21_review'], (*earlier, review))
    reviewed_radial_definition_hashes(prior['v21_review'], (*earlier, review))
    prior_tools = {item['path']: item['sha256'] for item in prior['v24_release_review']['validation_tools']}
    tools = review.get('validation_tools', ())
    if [item.get('path') for item in tools] != list(prior_tools):
        raise RuntimeError('v24.0.2 validation-tool review has missing or duplicate paths')
    for item in tools:
        value = item.get('sha256')
        if (item.get('previous_sha256') != prior_tools[item['path']] or not item.get('reason')
                or not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)):
            raise RuntimeError('v24.0.2 validation-tool review has invalid predecessor, digest or reason')
    return review


def reviewed_v24_0_3_release_contract(manifest, v21, *earlier_patches):
    """Authenticate TTA optimization against the tagged 24.0.2 receipt."""
    manifest = _without_reviewed_v24_0_4_release(manifest)
    key = 'v24_0_3_release_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V24_0_3_RELEASE_PREDECESSOR_SHA256:
        raise RuntimeError('v24.0.3 predecessor inventory changed; preserve every historical record')
    _authenticate_successor_once(prior, 'v24_0_2_release_review', reviewed_v24_0_2_release_contract)
    earlier = tuple(value for name, value in prior.items()
                    if name != 'v21_review' and isinstance(value, dict)
                    and 'release' in value and 'definitions' in value)
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='24.0.3',
        expected_digest=REVIEWED_V24_0_3_RELEASE_SHA256,
        previous_digest=REVIEWED_V24_0_2_RELEASE_SHA256,
        earlier_patches=earlier)
    if (review.get('predecessor_commit') != REVIEWED_V24_0_3_RELEASE_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V24_0_3_RELEASE_PREDECESSOR_SHA256
            or review.get('feature') != 'tta-throughput-restoration'):
        raise RuntimeError('v24.0.3 review has an unexpected predecessor or feature')
    snapshots = review.get('module_snapshots', ())
    modules = [item.get('module') for item in snapshots]
    if len(modules) != len(set(modules)) or set(modules) != set(REVIEWED_V24_0_3_RELEASE_PREDECESSOR_MODULES):
        raise RuntimeError('v24.0.3 source snapshot coverage differs')
    for item in snapshots:
        module = item['module']
        previous = REVIEWED_V24_0_3_RELEASE_PREDECESSOR_MODULES[module]
        historical = item.get('previous_top_level', ())
        historical_digest = hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest()
        if (item.get('previous_ast_sha256') != previous['ast_sha256']
                or historical_digest != previous['statements_sha256']):
            raise RuntimeError(f'v24.0.3 source predecessor changed: {module}')
        if item.get('removed'):
            raise RuntimeError(f'v24.0.3 has an unreviewed module removal: {module}')
        for value in (item.get('ast_sha256'), *historical, *item.get('top_level', ())):
            if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise RuntimeError(f'v24.0.3 source snapshot has an invalid digest: {module}')
        if not item.get('reason'):
            raise RuntimeError(f'v24.0.3 source snapshot has no review reason: {module}')
        positions = []
        for record in review['definitions'] + review['statements']:
            if record['module'] != module:
                continue
            current_index, previous_index = record.get('current_index'), record.get('previous_index')
            if (type(current_index) is not int or not 0 <= current_index < len(item['top_level'])
                    or item['top_level'][current_index] != record['sha256']):
                raise RuntimeError(f'v24.0.3 statement position differs: {module}')
            if previous_index is None:
                if record['previous_sha256'] is not None:
                    raise RuntimeError(f'v24.0.3 new statement has an unexpected predecessor: {module}')
            elif (type(previous_index) is not int or not 0 <= previous_index < len(historical)
                    or historical[previous_index] != record['previous_sha256']):
                raise RuntimeError(f'v24.0.3 statement predecessor changed: {module}')
            positions.append(current_index)
        if len(positions) != len(set(positions)):
            raise RuntimeError(f'v24.0.3 source has duplicate reviewed positions: {module}')
    if any(record['module'] not in set(modules) for record in review['definitions'] + review['statements']):
        raise RuntimeError('v24.0.3 statement has no complete source snapshot')
    if review.get('removed_definitions', ()) or review.get('removed_statements', ()):
        raise RuntimeError('v24.0.3 has an unreviewed statement removal')
    reviewed_radial_module_hashes(prior['v21_review'], (*earlier, review))
    reviewed_radial_definition_hashes(prior['v21_review'], (*earlier, review))
    prior_tools = {item['path']: item['sha256'] for item in prior['v24_0_2_release_review']['validation_tools']}
    tools = review.get('validation_tools', ())
    expected_paths = [*prior_tools, 'tools/qualify_radial_bitset_compaction.py',
                      'tools/qualify_d1_confidence_masked_transfer.py', 'tools/qualify_release.py']
    if [item.get('path') for item in tools] != expected_paths:
        raise RuntimeError('v24.0.3 validation-tool review has missing or duplicate paths')
    for item in tools:
        value = item.get('sha256')
        if (item.get('previous_sha256') != prior_tools.get(item['path']) or not item.get('reason')
                or not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)):
            raise RuntimeError('v24.0.3 validation-tool review has invalid predecessor, digest or reason')
    return review


def reviewed_v24_0_4_release_contract(manifest, v21, *earlier_patches):
    """Authenticate the cleanup appendix against the complete tagged 24.0.3 receipt."""
    manifest = _without_reviewed_v24_0_5_release(manifest)
    key = 'v24_0_4_release_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V24_0_4_RELEASE_PREDECESSOR_SHA256:
        raise RuntimeError('v24.0.4 predecessor inventory changed; preserve every historical record')
    _authenticate_successor_once(prior, 'v24_0_3_release_review', reviewed_v24_0_3_release_contract)
    earlier = tuple(value for name, value in prior.items()
                    if name != 'v21_review' and isinstance(value, dict)
                    and 'release' in value and 'definitions' in value)
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='24.0.4',
        expected_digest=REVIEWED_V24_0_4_RELEASE_SHA256,
        previous_digest=REVIEWED_V24_0_3_RELEASE_SHA256,
        earlier_patches=earlier)
    if (review.get('predecessor_commit') != REVIEWED_V24_0_4_RELEASE_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V24_0_4_RELEASE_PREDECESSOR_SHA256
            or review.get('feature') != 'legacy-configuration-and-compiled-cpu-policy'):
        raise RuntimeError('v24.0.4 review has an unexpected predecessor or feature')

    snapshots = review.get('module_snapshots', ())
    modules = [item.get('module') for item in snapshots]
    if (len(modules) != len(set(modules))
            or set(modules) != set(REVIEWED_V24_0_4_RELEASE_PREDECESSOR_MODULES)):
        raise RuntimeError('v24.0.4 source snapshot coverage differs')
    if set(review.get('complete_modules', ())) - set(modules):
        raise RuntimeError('v24.0.4 complete source snapshot has no module')
    records = review['definitions'] + review['statements']
    for item in snapshots:
        module = item['module']
        previous = REVIEWED_V24_0_4_RELEASE_PREDECESSOR_MODULES[module]
        historical = item.get('previous_top_level', ())
        historical_digest = hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest()
        if (item.get('previous_ast_sha256') != previous['ast_sha256']
                or historical_digest != previous['statements_sha256']):
            raise RuntimeError(f'v24.0.4 source predecessor changed: {module}')
        if item.get('removed'):
            raise RuntimeError(f'v24.0.4 has an unreviewed module removal: {module}')
        for value in (item.get('ast_sha256'), *historical, *item.get('top_level', ())):
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in '0123456789abcdef' for char in value)):
                raise RuntimeError(f'v24.0.4 source snapshot has an invalid digest: {module}')
        if not item.get('reason'):
            raise RuntimeError(f'v24.0.4 source snapshot has no review reason: {module}')
        positions = []
        for record in records:
            if record['module'] != module:
                continue
            current_index, previous_index = record.get('current_index'), record.get('previous_index')
            if (type(current_index) is not int or not 0 <= current_index < len(item['top_level'])
                    or item['top_level'][current_index] != record['sha256']):
                raise RuntimeError(f'v24.0.4 statement position differs: {module}')
            if previous_index is None:
                if record['previous_sha256'] is not None:
                    raise RuntimeError(f'v24.0.4 new statement has an unexpected predecessor: {module}')
            elif (type(previous_index) is not int or not 0 <= previous_index < len(historical)
                    or historical[previous_index] != record['previous_sha256']):
                raise RuntimeError(f'v24.0.4 statement predecessor changed: {module}')
            positions.append(current_index)
        if len(positions) != len(set(positions)):
            raise RuntimeError(f'v24.0.4 source has duplicate reviewed positions: {module}')
    if any(record['module'] not in set(modules) for record in records):
        raise RuntimeError('v24.0.4 statement has no complete source snapshot')

    removals = {
        'definitions': tuple(sorted((item['module'], item['name'], item['previous_index'],
                                     item['previous_sha256']) for item in review.get('removed_definitions', ()))),
        'statements': tuple(sorted((item['module'], item['previous_index'], item['previous_sha256'])
                                   for item in review.get('removed_statements', ()))),
    }
    if removals != REVIEWED_V24_0_4_RELEASE_REMOVALS:
        raise RuntimeError('v24.0.4 reviewed removal scope differs')
    removed_positions = set()
    for category in ('removed_definitions', 'removed_statements'):
        for record in review.get(category, ()):
            module, index = record['module'], record['previous_index']
            snapshot = next((item for item in snapshots if item['module'] == module), None)
            if (snapshot is None or type(index) is not int
                    or not 0 <= index < len(snapshot['previous_top_level'])
                    or snapshot['previous_top_level'][index] != record['previous_sha256']
                    or not record.get('reason') or (module, index) in removed_positions):
                raise RuntimeError(f'v24.0.4 retired predecessor changed: {module}')
            removed_positions.add((module, index))
    reviewed_radial_module_hashes(prior['v21_review'], (*earlier, review))
    reviewed_radial_definition_hashes(prior['v21_review'], (*earlier, review))

    prior_tools = {item['path']: item['sha256'] for item in prior['v24_0_3_release_review']['validation_tools']}
    tools = review.get('validation_tools', ())
    if [item.get('path') for item in tools] != list(prior_tools):
        raise RuntimeError('v24.0.4 validation-tool review has missing or duplicate paths')
    for item in tools:
        value = item.get('sha256')
        if (item.get('previous_sha256') != prior_tools.get(item['path']) or not item.get('reason')
                or not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)):
            raise RuntimeError('v24.0.4 validation-tool review has invalid predecessor, digest or reason')
    return review


def reviewed_v24_0_5_release_contract(manifest, v21, *earlier_patches):
    """Authenticate the throughput appendix against the complete tagged 24.0.4 receipt."""
    key = 'v24_0_5_release_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V24_0_5_RELEASE_PREDECESSOR_SHA256:
        raise RuntimeError('v24.0.5 predecessor inventory changed; preserve every historical record')
    _authenticate_successor_once(prior, 'v24_0_4_release_review', reviewed_v24_0_4_release_contract)
    earlier = tuple(value for name, value in prior.items()
                    if name != 'v21_review' and isinstance(value, dict)
                    and 'release' in value and 'definitions' in value)
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='24.0.5',
        expected_digest=REVIEWED_V24_0_5_RELEASE_SHA256,
        previous_digest=REVIEWED_V24_0_4_RELEASE_SHA256,
        earlier_patches=earlier)
    if (review.get('predecessor_commit') != REVIEWED_V24_0_5_RELEASE_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V24_0_5_RELEASE_PREDECESSOR_SHA256
            or review.get('feature') != 'tta-pta-throughput'):
        raise RuntimeError('v24.0.5 review has an unexpected predecessor or feature')

    snapshots = review.get('module_snapshots', ())
    modules = [item.get('module') for item in snapshots]
    if (len(modules) != len(set(modules))
            or set(modules) != set(REVIEWED_V24_0_5_RELEASE_PREDECESSOR_MODULES)):
        raise RuntimeError('v24.0.5 source snapshot coverage differs')
    if set(review.get('complete_modules', ())) - set(modules):
        raise RuntimeError('v24.0.5 complete source snapshot has no module')
    records = review['definitions'] + review['statements']
    for item in snapshots:
        module = item['module']
        previous = REVIEWED_V24_0_5_RELEASE_PREDECESSOR_MODULES[module]
        historical = item.get('previous_top_level', ())
        historical_digest = hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest()
        if (item.get('previous_ast_sha256') != previous['ast_sha256']
                or historical_digest != previous['statements_sha256']):
            raise RuntimeError(f'v24.0.5 source predecessor changed: {module}')
        if item.get('removed'):
            raise RuntimeError(f'v24.0.5 has an unreviewed module removal: {module}')
        for value in (item.get('ast_sha256'), *historical, *item.get('top_level', ())):
            if (not isinstance(value, str) or len(value) != 64
                    or any(char not in '0123456789abcdef' for char in value)):
                raise RuntimeError(f'v24.0.5 source snapshot has an invalid digest: {module}')
        if not item.get('reason'):
            raise RuntimeError(f'v24.0.5 source snapshot has no review reason: {module}')
        positions = []
        for record in records:
            if record['module'] != module:
                continue
            current_index, previous_index = record.get('current_index'), record.get('previous_index')
            if (type(current_index) is not int or not 0 <= current_index < len(item['top_level'])
                    or item['top_level'][current_index] != record['sha256']):
                raise RuntimeError(f'v24.0.5 statement position differs: {module}')
            if previous_index is None:
                if record['previous_sha256'] is not None:
                    raise RuntimeError(f'v24.0.5 new statement has an unexpected predecessor: {module}')
            elif (type(previous_index) is not int or not 0 <= previous_index < len(historical)
                    or historical[previous_index] != record['previous_sha256']):
                raise RuntimeError(f'v24.0.5 statement predecessor changed: {module}')
            positions.append(current_index)
        if len(positions) != len(set(positions)):
            raise RuntimeError(f'v24.0.5 source has duplicate reviewed positions: {module}')
    if any(record['module'] not in set(modules) for record in records):
        raise RuntimeError('v24.0.5 statement has no complete source snapshot')

    removals = {
        'definitions': tuple(sorted((item['module'], item['name'], item['previous_index'],
                                     item['previous_sha256']) for item in review.get('removed_definitions', ()))),
        'statements': tuple(sorted((item['module'], item['previous_index'], item['previous_sha256'])
                                   for item in review.get('removed_statements', ()))),
    }
    if removals != REVIEWED_V24_0_5_RELEASE_REMOVALS:
        raise RuntimeError('v24.0.5 reviewed removal scope differs')
    removed_positions = set()
    for category in ('removed_definitions', 'removed_statements'):
        for record in review.get(category, ()):
            module, index = record['module'], record['previous_index']
            snapshot = next((item for item in snapshots if item['module'] == module), None)
            if (snapshot is None or type(index) is not int
                    or not 0 <= index < len(snapshot['previous_top_level'])
                    or snapshot['previous_top_level'][index] != record['previous_sha256']
                    or not record.get('reason') or (module, index) in removed_positions):
                raise RuntimeError(f'v24.0.5 retired predecessor changed: {module}')
            removed_positions.add((module, index))
    reviewed_radial_module_hashes(prior['v21_review'], (*earlier, review))
    reviewed_radial_definition_hashes(prior['v21_review'], (*earlier, review))

    prior_tools = {item['path']: item['sha256'] for item in prior['v24_0_4_release_review']['validation_tools']}
    tools = review.get('validation_tools', ())
    expected_paths = [*prior_tools, *REVIEWED_V24_0_5_ADDED_VALIDATION_TOOLS]
    if [item.get('path') for item in tools] != expected_paths:
        raise RuntimeError('v24.0.5 validation-tool review has missing or duplicate paths')
    for item in tools:
        value = item.get('sha256')
        previous = prior_tools.get(item['path'], REVIEWED_V24_0_5_ADDED_VALIDATION_TOOLS.get(item['path']))
        if (item.get('previous_sha256') != previous or not item.get('reason')
                or not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)):
            raise RuntimeError('v24.0.5 validation-tool review has invalid predecessor, digest or reason')
    return review


def _without_reviewed_v24_0_5_release(manifest):
    key = 'v24_0_5_release_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v24_0_5_release_contract)
    return {name: value for name, value in manifest.items() if name != key}


def _without_reviewed_v24_0_4_release(manifest):
    manifest = _without_reviewed_v24_0_5_release(manifest)
    key = 'v24_0_4_release_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v24_0_4_release_contract)
    return {name: value for name, value in manifest.items() if name != key}


def _without_reviewed_v24_0_3_release(manifest):
    manifest = _without_reviewed_v24_0_4_release(manifest)
    key = 'v24_0_3_release_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v24_0_3_release_contract)
    return {name: value for name, value in manifest.items() if name != key}


def _without_reviewed_v24_0_2_release(manifest):
    manifest = _without_reviewed_v24_0_3_release(manifest)
    key = 'v24_0_2_release_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v24_0_2_release_contract)
    return {name: value for name, value in manifest.items() if name != key}


def _without_reviewed_v24_release(manifest):
    """Admit the authenticated semantic successor before historical keyset checks."""
    manifest = _without_reviewed_v24_0_2_release(manifest)
    key = 'v24_release_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v24_release_contract)
    return {name: value for name, value in manifest.items() if name != key}


def reviewed_v22_3_2_release_contract(manifest, v21, *earlier_patches):
    """Authenticate bounded publication changes against the complete prior release."""
    manifest = _without_reviewed_v24_release(manifest)
    key = 'v22_3_2_release_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_3_2_RELEASE_PREDECESSOR_SHA256:
        raise RuntimeError('v22.3.2 predecessor inventory changed; preserve every historical record')
    ordered_keys = (
        *(f'v21_0_{index}_review' for index in range(1, 7)),
        'v21_1_review', 'v21_1_1_review', 'v21_1_2_review',
        'v22_augmentation_review', 'v22_coverage_review', 'v22_release_review',
        'v22_policy_memory_review', 'v22_policy_throughput_review',
        'v22_radial_retirement_review', 'v22_policy_window_review',
        'v22_tilted_azimuthal_gpu_review', 'v22_pta_throughput_review', 'v22_1_release_review',
        'v22_2_release_review', 'v22_3_release_review', 'v22_3_1_release_review',
    )
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='22.3.2',
        expected_digest=REVIEWED_V22_3_2_RELEASE_SHA256,
        previous_digest=REVIEWED_V22_3_1_RELEASE_SHA256,
        earlier_patches=tuple(prior[name] for name in ordered_keys),
    )
    if (review.get('predecessor_commit') != REVIEWED_V22_3_2_RELEASE_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_3_2_RELEASE_PREDECESSOR_SHA256
            or review.get('feature') != 'bounded-confidence-publication-throughput'):
        raise RuntimeError('v22.3.2 review has an unexpected predecessor or source scope')
    snapshots = review.get('module_snapshots', ())
    modules = [item.get('module') for item in snapshots]
    if len(modules) != len(set(modules)) or set(modules) != set(REVIEWED_V22_3_2_RELEASE_PREDECESSOR_MODULES):
        raise RuntimeError('v22.3.2 source snapshot coverage differs')
    for item in snapshots:
        module = item['module']
        previous = REVIEWED_V22_3_2_RELEASE_PREDECESSOR_MODULES[module]
        historical = item.get('previous_top_level', ())
        historical_digest = hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest()
        if (item.get('previous_ast_sha256') != previous['ast_sha256']
                or historical_digest != previous['statements_sha256']):
            raise RuntimeError(f'v22.3.2 source predecessor changed: {module}')
        if 'removed' in item:
            raise RuntimeError(f'v22.3.2 has an unreviewed module removal: {module}')
        for value in (item.get('ast_sha256'), *historical, *item.get('top_level', ())):
            if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise RuntimeError(f'v22.3.2 source snapshot has an invalid digest: {module}')
        if not item.get('reason'):
            raise RuntimeError(f'v22.3.2 source snapshot has no review reason: {module}')
        positions = []
        for record in review['definitions'] + review['statements']:
            if record['module'] != module:
                continue
            current_index = record.get('current_index')
            previous_index = record.get('previous_index')
            if (type(current_index) is not int or not 0 <= current_index < len(item['top_level'])
                    or item['top_level'][current_index] != record['sha256']):
                raise RuntimeError(f'v22.3.2 statement position differs: {module}')
            if previous_index is None:
                if record['previous_sha256'] is not None:
                    raise RuntimeError(f'v22.3.2 new statement has an unexpected predecessor: {module}')
            elif (type(previous_index) is not int or not 0 <= previous_index < len(historical)
                    or historical[previous_index] != record['previous_sha256']):
                raise RuntimeError(f'v22.3.2 statement predecessor changed: {module}')
            positions.append(current_index)
        if len(positions) != len(set(positions)):
            raise RuntimeError(f'v22.3.2 source has duplicate reviewed positions: {module}')
    if any(record['module'] not in set(modules) for record in review['definitions'] + review['statements']):
        raise RuntimeError('v22.3.2 statement has no complete source snapshot')
    reviewed_radial_module_hashes(prior['v21_review'], (*tuple(prior[name] for name in ordered_keys), review))
    reviewed_radial_definition_hashes(prior['v21_review'], (*tuple(prior[name] for name in ordered_keys), review))
    validation_tools = review.get('validation_tools', ())
    if [item.get('path') for item in validation_tools] != [
            'tools/compare_reconciliation.py', 'tools/qualify_tta_reconciliation.py',
            'tools/export_reconciliation_evidence.py', 'tools/qualify_confidence_consolidation.py',
            'tools/qualify_d1_confidence_bounds.py', 'tools/analyze_pipeline_trace.py']:
        raise RuntimeError('v22.3.2 validation-tool review has missing or duplicate paths')
    prior_tools = {item['path']: item['sha256'] for item in prior['v22_3_1_release_review']['validation_tools']}
    for item in validation_tools:
        if item.get('previous_sha256') != prior_tools.get(item['path']):
            raise RuntimeError('v22.3.2 validation-tool predecessor changed')
        value = item.get('sha256')
        if (not item.get('reason') or not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)):
            raise RuntimeError('v22.3.2 validation-tool review has an invalid digest or reason')
    return review


def _without_reviewed_v22_3_2_release(manifest):
    """Admit the authenticated publication successor before older keyset checks."""
    manifest = _without_reviewed_v24_release(manifest)
    key = 'v22_3_2_release_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v22_3_2_release_contract)
    return {name: value for name, value in manifest.items() if name != key}


def reviewed_v22_3_1_release_contract(manifest, v21, *earlier_patches):
    """Authenticate additions against the complete released inventory and sources."""
    manifest = _without_reviewed_v22_3_2_release(manifest)
    key = 'v22_3_1_release_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_3_1_RELEASE_PREDECESSOR_SHA256:
        raise RuntimeError('v22.3.1 predecessor inventory changed; preserve every historical record')
    ordered_keys = (
        *(f'v21_0_{index}_review' for index in range(1, 7)),
        'v21_1_review', 'v21_1_1_review', 'v21_1_2_review',
        'v22_augmentation_review', 'v22_coverage_review', 'v22_release_review',
        'v22_policy_memory_review', 'v22_policy_throughput_review',
        'v22_radial_retirement_review', 'v22_policy_window_review',
        'v22_tilted_azimuthal_gpu_review', 'v22_pta_throughput_review', 'v22_1_release_review',
        'v22_2_release_review', 'v22_3_release_review',
    )
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='22.3.1',
        expected_digest=REVIEWED_V22_3_1_RELEASE_SHA256,
        previous_digest=REVIEWED_V22_3_RELEASE_SHA256,
        earlier_patches=tuple(prior[name] for name in ordered_keys),
    )
    if (review.get('predecessor_commit') != REVIEWED_V22_3_1_RELEASE_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_3_1_RELEASE_PREDECESSOR_SHA256
            or review.get('feature') != 'efficient-union-and-native-confidence'):
        raise RuntimeError('v22.3.1 review has an unexpected predecessor or source scope')
    snapshots = review.get('module_snapshots', ())
    modules = [item.get('module') for item in snapshots]
    if len(modules) != len(set(modules)) or set(modules) != set(REVIEWED_V22_3_1_RELEASE_PREDECESSOR_MODULES):
        raise RuntimeError('v22.3.1 source snapshot coverage differs')
    removed = {item['module'] for item in snapshots if item.get('removed') is True}
    if removed != REVIEWED_V22_3_1_REMOVED_MODULES:
        raise RuntimeError('v22.3.1 reviewed module removals differ')
    for item in snapshots:
        module = item['module']
        previous = REVIEWED_V22_3_1_RELEASE_PREDECESSOR_MODULES[module]
        historical = item.get('previous_top_level', ())
        historical_digest = hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest()
        if (item.get('previous_ast_sha256') != previous['ast_sha256']
                or historical_digest != previous['statements_sha256']):
            raise RuntimeError(f'v22.3.1 source predecessor changed: {module}')
        if module in removed:
            if (item.get('ast_sha256') is not None or item.get('top_level') != []
                    or previous['ast_sha256'] is None
                    or module in review.get('complete_modules', ())
                    or any(record['module'] == module for category in (
                        'definitions', 'statements', 'local_import_seam_updates')
                        for record in review.get(category, ()))):
                raise RuntimeError(f'v22.3.1 removed module retains current source records: {module}')
            digests = historical
        else:
            if 'removed' in item:
                raise RuntimeError(f'v22.3.1 source snapshot has an invalid removal marker: {module}')
            digests = (item.get('ast_sha256'), *historical, *item.get('top_level', ()))
        for value in digests:
            if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise RuntimeError(f'v22.3.1 source snapshot has an invalid digest: {module}')
        if not item.get('reason'):
            raise RuntimeError(f'v22.3.1 source snapshot has no review reason: {module}')
        positions = []
        for record in review['definitions'] + review['statements']:
            if record['module'] != module:
                continue
            current_index = record.get('current_index')
            previous_index = record.get('previous_index')
            if (type(current_index) is not int or not 0 <= current_index < len(item['top_level'])
                    or item['top_level'][current_index] != record['sha256']):
                raise RuntimeError(f'v22.3.1 statement position differs: {module}')
            if previous_index is None:
                if record['previous_sha256'] is not None:
                    raise RuntimeError(f'v22.3.1 new statement has an unexpected predecessor: {module}')
            elif (type(previous_index) is not int or not 0 <= previous_index < len(historical)
                    or historical[previous_index] != record['previous_sha256']):
                raise RuntimeError(f'v22.3.1 statement predecessor changed: {module}')
            positions.append(current_index)
        if len(positions) != len(set(positions)):
            raise RuntimeError(f'v22.3.1 source has duplicate reviewed positions: {module}')
    if any(record['module'] not in set(modules) for record in review['definitions'] + review['statements']):
        raise RuntimeError('v22.3.1 statement has no complete source snapshot')
    reviewed_radial_module_hashes(prior['v21_review'], (*tuple(prior[name] for name in ordered_keys), review))
    reviewed_radial_definition_hashes(prior['v21_review'], (*tuple(prior[name] for name in ordered_keys), review))
    validation_tools = review.get('validation_tools', ())
    if [item.get('path') for item in validation_tools] != [
            'tools/compare_reconciliation.py', 'tools/qualify_tta_reconciliation.py',
            'tools/export_reconciliation_evidence.py']:
        raise RuntimeError('v22.3.1 validation-tool review has missing or duplicate paths')
    prior_tools = {item['path']: item['sha256'] for item in prior['v22_3_release_review']['validation_tools']}
    for item in validation_tools:
        if item.get('previous_sha256') != prior_tools.get(item['path']):
            raise RuntimeError('v22.3.1 validation-tool predecessor changed')
        value = item.get('sha256')
        if (not item.get('reason') or not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)):
            raise RuntimeError('v22.3.1 validation-tool review has an invalid digest or reason')
    return review


def _without_reviewed_v22_3_1_release(manifest):
    """Admit the authenticated performance successor before older keyset checks."""
    manifest = _without_reviewed_v22_3_2_release(manifest)
    key = 'v22_3_1_release_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v22_3_1_release_contract)
    return {name: value for name, value in manifest.items() if name != key}



def reviewed_v22_3_release_contract(manifest, v21, *earlier_patches):
    """Authenticate additions against the complete released inventory and sources."""
    manifest = _without_reviewed_v22_3_1_release(manifest)
    key = 'v22_3_release_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_3_RELEASE_PREDECESSOR_SHA256:
        raise RuntimeError('v22.3.0 predecessor inventory changed; preserve every historical record')
    ordered_keys = (
        *(f'v21_0_{index}_review' for index in range(1, 7)),
        'v21_1_review', 'v21_1_1_review', 'v21_1_2_review',
        'v22_augmentation_review', 'v22_coverage_review', 'v22_release_review',
        'v22_policy_memory_review', 'v22_policy_throughput_review',
        'v22_radial_retirement_review', 'v22_policy_window_review',
        'v22_tilted_azimuthal_gpu_review', 'v22_pta_throughput_review', 'v22_1_release_review',
        'v22_2_release_review',
    )
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='22.3.0',
        expected_digest=REVIEWED_V22_3_RELEASE_SHA256,
        previous_digest=REVIEWED_V22_2_RELEASE_SHA256,
        earlier_patches=tuple(prior[name] for name in ordered_keys),
    )
    if (review.get('predecessor_commit') != REVIEWED_V22_3_RELEASE_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_3_RELEASE_PREDECESSOR_SHA256
            or review.get('feature') != 'external-reconciliation-and-retained-confidence'):
        raise RuntimeError('v22.3.0 review has an unexpected predecessor or source scope')
    snapshots = review.get('module_snapshots', ())
    modules = [item.get('module') for item in snapshots]
    if len(modules) != len(set(modules)) or set(modules) != set(REVIEWED_V22_3_RELEASE_PREDECESSOR_MODULES):
        raise RuntimeError('v22.3.0 source snapshot coverage differs')
    for item in snapshots:
        module = item['module']
        previous = REVIEWED_V22_3_RELEASE_PREDECESSOR_MODULES[module]
        historical = item.get('previous_top_level', ())
        historical_digest = hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest()
        if (item.get('previous_ast_sha256') != previous['ast_sha256']
                or historical_digest != previous['statements_sha256']):
            raise RuntimeError(f'v22.3.0 source predecessor changed: {module}')
        for value in (item.get('ast_sha256'), *historical, *item.get('top_level', ())):
            if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise RuntimeError(f'v22.3.0 source snapshot has an invalid digest: {module}')
        if not item.get('reason'):
            raise RuntimeError(f'v22.3.0 source snapshot has no review reason: {module}')
        positions = []
        for record in review['definitions'] + review['statements']:
            if record['module'] != module:
                continue
            current_index = record.get('current_index')
            previous_index = record.get('previous_index')
            if (type(current_index) is not int or not 0 <= current_index < len(item['top_level'])
                    or item['top_level'][current_index] != record['sha256']):
                raise RuntimeError(f'v22.3.0 statement position differs: {module}')
            if previous_index is None:
                if record['previous_sha256'] is not None:
                    raise RuntimeError(f'v22.3.0 new statement has an unexpected predecessor: {module}')
            elif (type(previous_index) is not int or not 0 <= previous_index < len(historical)
                    or historical[previous_index] != record['previous_sha256']):
                raise RuntimeError(f'v22.3.0 statement predecessor changed: {module}')
            positions.append(current_index)
        if len(positions) != len(set(positions)):
            raise RuntimeError(f'v22.3.0 source has duplicate reviewed positions: {module}')
    if any(record['module'] not in set(modules) for record in review['definitions'] + review['statements']):
        raise RuntimeError('v22.3.0 statement has no complete source snapshot')
    reviewed_radial_module_hashes(prior['v21_review'], (*tuple(prior[name] for name in ordered_keys), review))
    reviewed_radial_definition_hashes(prior['v21_review'], (*tuple(prior[name] for name in ordered_keys), review))
    validation_tools = review.get('validation_tools', ())
    if [item.get('path') for item in validation_tools] != [
            'tools/compare_reconciliation.py', 'tools/qualify_tta_reconciliation.py']:
        raise RuntimeError('v22.3.0 validation-tool review has missing or duplicate paths')
    for item in validation_tools:
        value = item.get('sha256')
        if (not item.get('reason') or not isinstance(value, str) or len(value) != 64
                or any(char not in '0123456789abcdef' for char in value)):
            raise RuntimeError('v22.3.0 validation-tool review has an invalid digest or reason')
    return review


def _without_reviewed_v22_3_release(manifest):
    """Admit only the independently authenticated reconciliation successor."""
    manifest = _without_reviewed_v22_3_1_release(manifest)
    key = 'v22_3_release_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v22_3_release_contract)
    return {name: value for name, value in manifest.items() if name != key}


def verify_v22_3_source_snapshots(review, trees, successors=()):
    """Rewind authenticated successors before checking the release source snapshot."""
    for item in review['module_snapshots']:
        module = item['module']
        tree = trees.get(module)
        actual = digest(tree) if tree is not None else None
        for successor in reversed(successors):
            update = next((record for record in successor.get('module_snapshots', ())
                           if record['module'] == module), None)
            if update is not None:
                if actual != update['ast_sha256']:
                    raise RuntimeError(f'v{review["release"]} reviewed source changed: {module}')
                actual = update['previous_ast_sha256']
        if actual != item['ast_sha256']:
            raise RuntimeError(f'v{review["release"]} reviewed source changed: {module}')
        rewind_reviewed_statements(module, tree.body if tree is not None else (), (review, *successors))


def verify_v22_3_validation_tools(review, successors=()):
    """Authenticate tool sources through explicit successor hash links."""
    for item in review['validation_tools']:
        path = ROOT / item['path']
        expected = item['sha256']
        for successor in successors:
            update = next((record for record in successor.get('validation_tools', ())
                           if record['path'] == item['path']), None)
            if update is not None:
                if update.get('previous_sha256') != expected:
                    raise RuntimeError(f'v{review["release"]} validation tool changed: predecessor for {item["path"]}')
                expected = update['sha256']
        if (not path.is_file()
                or hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest() != expected):
            raise RuntimeError(f'v{review["release"]} validation tool changed or is missing: {item["path"]}')


def reviewed_v22_2_release_contract(manifest, v21, *earlier_patches):
    """Authenticate additions against the complete released inventory and sources."""
    manifest = _without_reviewed_v22_3_release(manifest)
    key = 'v22_2_release_review'
    prior = {name: value for name, value in manifest.items() if name != key}
    encoded = json.dumps(prior, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != REVIEWED_V22_2_RELEASE_PREDECESSOR_SHA256:
        raise RuntimeError('v22.2.0 predecessor inventory changed; preserve every historical record')
    ordered_keys = (
        *(f'v21_0_{index}_review' for index in range(1, 7)),
        'v21_1_review', 'v21_1_1_review', 'v21_1_2_review',
        'v22_augmentation_review', 'v22_coverage_review', 'v22_release_review',
        'v22_policy_memory_review', 'v22_policy_throughput_review',
        'v22_radial_retirement_review', 'v22_policy_window_review',
        'v22_tilted_azimuthal_gpu_review', 'v22_pta_throughput_review', 'v22_1_release_review',
    )
    review = _reviewed_v21_patch_contract(
        manifest, prior['v21_review'], key=key, release='22.2.0',
        expected_digest=REVIEWED_V22_2_RELEASE_SHA256,
        previous_digest=REVIEWED_V22_1_RELEASE_SHA256,
        earlier_patches=tuple(prior[name] for name in ordered_keys),
    )
    if (review.get('predecessor_commit') != REVIEWED_V22_2_RELEASE_PREDECESSOR_COMMIT
            or review.get('predecessor_inventory_sha256') != REVIEWED_V22_2_RELEASE_PREDECESSOR_SHA256
            or review.get('feature') != 'backend-policies-certified-coverage-and-pta-binary'
            or any(review.get(category) for category in (
                'preserved_radial_definition_updates', 'preserved_radial_module_updates'))):
        raise RuntimeError('v22.2.0 review has an unexpected predecessor or source scope')
    snapshots = review.get('module_snapshots', ())
    modules = [item.get('module') for item in snapshots]
    if len(modules) != len(set(modules)) or set(modules) != set(REVIEWED_V22_2_RELEASE_PREDECESSOR_MODULES):
        raise RuntimeError('v22.2.0 source snapshot coverage differs')
    for item in snapshots:
        module = item['module']
        previous = REVIEWED_V22_2_RELEASE_PREDECESSOR_MODULES[module]
        historical = item.get('previous_top_level', ())
        historical_digest = hashlib.sha256(json.dumps(historical, separators=(',', ':')).encode()).hexdigest()
        if (item.get('previous_ast_sha256') != previous['ast_sha256']
                or historical_digest != previous['statements_sha256']):
            raise RuntimeError(f'v22.2.0 source predecessor changed: {module}')
        for value in (item.get('ast_sha256'), *historical, *item.get('top_level', ())):
            if not isinstance(value, str) or len(value) != 64 or any(char not in '0123456789abcdef' for char in value):
                raise RuntimeError(f'v22.2.0 source snapshot has an invalid digest: {module}')
        if not item.get('reason'):
            raise RuntimeError(f'v22.2.0 source snapshot has no review reason: {module}')
        positions = []
        for record in review['definitions'] + review['statements']:
            if record['module'] != module:
                continue
            current_index = record.get('current_index')
            previous_index = record.get('previous_index')
            if (type(current_index) is not int or not 0 <= current_index < len(item['top_level'])
                    or item['top_level'][current_index] != record['sha256']):
                raise RuntimeError(f'v22.2.0 statement position differs: {module}')
            if previous_index is None:
                if record['previous_sha256'] is not None:
                    raise RuntimeError(f'v22.2.0 new statement has an unexpected predecessor: {module}')
            elif (type(previous_index) is not int or not 0 <= previous_index < len(historical)
                    or historical[previous_index] != record['previous_sha256']):
                raise RuntimeError(f'v22.2.0 statement predecessor changed: {module}')
            positions.append(current_index)
        if len(positions) != len(set(positions)):
            raise RuntimeError(f'v22.2.0 source has duplicate reviewed positions: {module}')
    if any(record['module'] not in set(modules) for record in review['definitions'] + review['statements']):
        raise RuntimeError('v22.2.0 statement has no complete source snapshot')
    return review


def _without_reviewed_v22_2_release(manifest):
    """Historical keysets admit only the independently authenticated successor."""
    manifest = _without_reviewed_v22_3_release(manifest)
    key = 'v22_2_release_review'
    if key not in manifest:
        return manifest
    _authenticate_successor_once(manifest, key, reviewed_v22_2_release_contract)
    return {name: value for name, value in manifest.items() if name != key}


def rewind_reviewed_statements(module, statements, successors):
    """Restore authenticated predecessor hashes while retaining statement identities."""
    current = [(node, digest(node)) for node in statements]
    for successor in reversed(successors):
        snapshot = next((item for item in successor.get('module_snapshots', ()) if item['module'] == module), None)
        records = [item for category in ('definitions', 'statements')
                   for item in successor[category] if item['module'] == module]
        by_hash = {item['sha256']: item for item in records}
        if snapshot is not None and 'top_level' in snapshot:
            if [value for _, value in current] != snapshot['top_level']:
                raise RuntimeError(f'reviewed source statements changed: {module}')
        if snapshot is not None and snapshot.get('removed') is True:
            if current or records or snapshot.get('ast_sha256') is not None:
                raise RuntimeError(f'reviewed removed module has current source: {module}')
            current = [(None, value) for value in snapshot['previous_top_level']]
            continue
        restored = []
        for node, value in current:
            record = by_hash.get(value)
            if record is None:
                restored.append((node, value))
            elif record.get('previous_sha256') is not None:
                restored.append((node, record['previous_sha256']))
        retired_records = (*successor.get('removed_definitions', ()),
                           *successor.get('removed_statements', ()))
        for retired in sorted((item for item in retired_records if item['module'] == module),
                              key=lambda item: item['previous_index']):
            restored.insert(retired['previous_index'], (None, retired['previous_sha256']))
        current = restored
        if snapshot is not None and 'previous_top_level' in snapshot:
            if [value for _, value in current] != snapshot['previous_top_level']:
                raise RuntimeError(f'reviewed predecessor statement coverage differs: {module}')
    return current


def verify_v22_2_source_snapshots(review, trees, successors=()):
    """Rewind authenticated successors before checking the complete v22.2 sources."""
    for item in review['module_snapshots']:
        module = item['module']
        actual = digest(trees[module])
        for successor in reversed(successors):
            update = next((record for record in successor.get('module_snapshots', ())
                           if record['module'] == module), None)
            if update is not None:
                if actual != update['ast_sha256']:
                    raise RuntimeError(f'v22.2.0 reviewed source changed: {module}')
                actual = update['previous_ast_sha256']
        if actual != item['ast_sha256']:
            raise RuntimeError(f'v22.2.0 reviewed source changed: {module}')
        rewind_reviewed_statements(module, trees[module].body, (review, *successors))


def verify_v22_1_release_scope(review, top_level, successors=()) -> None:
    """Every other AST statement in the three version-bearing modules is fixed."""
    bindings = {}
    for item in review['statements']:
        bindings.setdefault(item['module'], set()).add(item['binding'])
    for module, expected in REVIEWED_V22_1_RELEASE_PRESERVED_MODULES.items():
        try:
            restored = rewind_reviewed_statements(module, top_level[module], successors)
        except RuntimeError as exc:
            raise RuntimeError(f'v22.1.0 release changed non-version module statements: {module}') from exc
        hashes = [value for node, value in restored
                  if not (isinstance(node, ast.Assign) and any(
                      isinstance(target, ast.Name) and target.id in bindings[module]
                      for target in node.targets))]
        actual = hashlib.sha256(json.dumps(hashes, separators=(',', ':')).encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'v22.1.0 release changed non-version module statements: {module}')


def verify_pta_source_snapshots(review, trees, successors=()) -> None:
    """Cover entire existing PTA modules in addition to changed named records."""
    for item in review['module_snapshots']:
        actual = digest(trees[item['module']])
        for successor in reversed(successors):
            update = next((record for record in successor.get('module_snapshots', ())
                           if record['module'] == item['module']), None)
            if update is not None:
                if actual != update['ast_sha256']:
                    raise RuntimeError(f"v22 PTA throughput complete source changed: {item['module']}")
                actual = update['previous_ast_sha256']
        if actual != item['ast_sha256']:
            raise RuntimeError(f"v22 PTA throughput complete source changed: {item['module']}")


def verify_policy_window_runtime_scope(review, pipeline_statements, successors=()) -> None:
    """Reconstruct the window review's source snapshot through approved successors."""
    try:
        restored = rewind_reviewed_statements('pipeline', pipeline_statements, successors)
    except RuntimeError as exc:
        raise RuntimeError('v22 policy window changed unreviewed pipeline statements or imports') from exc
    preserved = [value for node, value in restored if getattr(node, 'name', None) != '_main_impl']
    encoded = json.dumps(preserved, separators=(',', ':')).encode('utf-8')
    if hashlib.sha256(encoded).hexdigest() != review['preserved_pipeline_statements_sha256']:
        raise RuntimeError('v22 policy window changed unreviewed pipeline statements or imports')


def reviewed_v20_statement_hashes(reviews) -> dict[tuple[str, str], str]:
    """Resolve authenticated successors without replacing the original v20 pins."""
    expected = {key: record[0] for key, record in REVIEWED_V20_ADDED_STATEMENTS.items()}
    for review in reviews:
        for item in review['statements']:
            key = (item['module'], item['label'])
            if key not in expected:
                continue
            if item.get('previous_sha256') != expected[key]:
                raise RuntimeError(f'v20 statement successor does not match its historical pin: {key[0]}.{key[1]}')
            expected[key] = item['sha256']
    return expected


def verify_augmentation_relocations(review, top_level) -> None:
    """Verify exact shared definitions and the original owner's public imports."""
    for item in review['definition_relocations']:
        module, name = item['module'], item['name']
        source = top_level[module]
        destination = top_level[item['destination_module']]
        matches = [node for node in destination if getattr(node, 'name', None) == item['destination_name']]
        reexports = [
            node for node in source if isinstance(node, ast.ImportFrom)
            and node.level == 1 and node.module == item['destination_module']
            and any(alias.name == name and alias.asname in (None, name) for alias in node.names)
        ]
        if (any(getattr(node, 'name', None) == name for node in source)
                or len(matches) != 1 or digest(matches[0]) != item['sha256']
                or len(reexports) != 1 or digest(reexports[0]) != item['reexport_sha256']):
            raise RuntimeError(f'v22 augmentation shared-owner relocation changed: {module}.{name}')


def reviewed_radial_module_hashes(v21, patches):
    """Require explicit authenticated successors for preserved full modules."""
    expected = {item['module']: item['sha256'] for item in v21['preserved_radial_modules']}
    for patch in patches:
        seen = set()
        for item in patch.get('preserved_radial_module_updates', ()):
            module = item.get('module')
            if module in seen or module not in expected or not item.get('reason'):
                raise RuntimeError('unknown, duplicate or unexplained Radial module review')
            seen.add(module)
            if item.get('previous_sha256') != expected[module]:
                raise RuntimeError('Radial module review does not match its preserved predecessor')
            value = item.get('sha256')
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise RuntimeError('Radial module review has an invalid digest')
            expected[module] = value
    return expected


def reviewed_radial_definition_hashes(v21, patches):
    """Permit only authenticated, predecessor-pinned updates to preserved methods."""
    expected = {(item['module'], item['qualified_name']): item['sha256']
                for item in v21['preserved_radial_definitions']}
    expected[('cylindrical_cuda_projection', 'RadialCudaProjector._upload_cropped_source')] = (
        REVIEWED_PRESERVED_RADIAL_UPLOAD_SHA256)
    for patch in patches:
        seen = set()
        for item in patch.get('preserved_radial_definition_updates', ()):
            key = (item['module'], item['qualified_name'])
            if key in seen or key not in expected or not item.get('reason'):
                raise RuntimeError('unknown, duplicate or unexplained Radial method review')
            seen.add(key)
            if item.get('previous_sha256') != expected[key]:
                raise RuntimeError('Radial method review does not match its preserved predecessor')
            value = item.get('sha256')
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise RuntimeError('Radial method review has an invalid digest')
            expected[key] = value
    return expected


def _verify_main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    baseline_digest = hashlib.sha256(json.dumps(
        manifest['statements'], sort_keys=True, separators=(',', ':'),
    ).encode('utf-8')).hexdigest()
    if baseline_digest != IMMUTABLE_INVENTORY_STATEMENTS_SHA256:
        raise RuntimeError('immutable inventory digest mismatch; retain historical statement records and add explicit reviews')
    rename_replacements = azimuthal_rename_replacements(manifest)
    v21 = reviewed_v21_contract(manifest)
    patch = reviewed_v21_patch_contract(manifest, v21)
    second_patch = reviewed_v21_0_2_contract(manifest, v21, patch)
    third_patch = reviewed_v21_0_3_contract(manifest, v21, patch, second_patch)
    fourth_patch = reviewed_v21_0_4_contract(manifest, v21, patch, second_patch, third_patch)
    fifth_patch = reviewed_v21_0_5_contract(manifest, v21, patch, second_patch, third_patch, fourth_patch)
    sixth_patch = reviewed_v21_0_6_contract(manifest, v21, patch, second_patch, third_patch, fourth_patch, fifth_patch)
    patches = (patch, second_patch, third_patch, fourth_patch, fifth_patch, sixth_patch)
    lta_release = reviewed_v21_1_contract(manifest, v21, *patches)
    patches = (*patches, lta_release)
    overlap_release = reviewed_v21_1_1_contract(manifest, v21, *patches)
    patches = (*patches, overlap_release)
    frontier_release = reviewed_v21_1_2_contract(manifest, v21, *patches)

    augmentation_review = reviewed_v22_augmentation_contract(manifest, v21, *patches)
    coverage_review = reviewed_v22_coverage_contract(manifest, v21, *patches, augmentation_review)
    verify_coverage_validation_tools(coverage_review)
    # Validate each fork against its own predecessor, then join their disjoint
    # runtime changes. Release identity successors follow the main parent.
    patches = (*patches, frontier_release, augmentation_review, coverage_review)
    release_review = reviewed_v22_release_contract(manifest, v21, *patches)
    patches = (*patches, release_review)
    memory_review = reviewed_v22_policy_memory_contract(manifest, v21, *patches)
    patches = (*patches, memory_review)
    throughput_review = reviewed_v22_policy_throughput_contract(manifest, v21, *patches)
    patches = (*patches, throughput_review)
    radial_retirement_review = reviewed_v22_radial_retirement_contract(manifest, v21, *patches)
    patches = (*patches, radial_retirement_review)
    policy_window_review = reviewed_v22_policy_window_contract(manifest, v21, *patches)
    patches = (*patches, policy_window_review)
    tilted_azimuthal_review = reviewed_v22_tilted_azimuthal_gpu_contract(manifest, v21, *patches)
    patches = (*patches, tilted_azimuthal_review)
    pta_throughput_review = reviewed_v22_pta_throughput_contract(manifest, v21, *patches)
    patches = (*patches, pta_throughput_review)
    version_release_review = reviewed_v22_1_release_contract(manifest, v21, *patches)
    patches = (*patches, version_release_review)
    backend_release_review = reviewed_v22_2_release_contract(manifest, v21, *patches)
    patches = (*patches, backend_release_review)
    reconciliation_release_review = (reviewed_v22_3_release_contract(manifest, v21, *patches)
                                     if 'v22_3_release_review' in manifest else None)
    performance_release_review = (reviewed_v22_3_1_release_contract(manifest, v21)
                                  if 'v22_3_1_release_review' in manifest else None)
    publication_release_review = (reviewed_v22_3_2_release_contract(manifest, v21)
                                  if 'v22_3_2_release_review' in manifest else None)
    throughput_release_review = (_authenticate_successor_once(
                                 manifest, 'v24_0_5_release_review', reviewed_v24_0_5_release_contract)
                                 if 'v24_0_5_release_review' in manifest else None)
    throughput_manifest = _without_reviewed_v24_0_5_release(manifest)
    cleanup_release_review = (_authenticate_successor_once(
                              throughput_manifest, 'v24_0_4_release_review', reviewed_v24_0_4_release_contract)
                              if 'v24_0_4_release_review' in manifest else None)
    cleanup_manifest = _without_reviewed_v24_0_4_release(throughput_manifest)
    tta_release_review = (_authenticate_successor_once(
                          cleanup_manifest, 'v24_0_3_release_review', reviewed_v24_0_3_release_contract)
                          if 'v24_0_3_release_review' in manifest else None)
    patch_manifest = _without_reviewed_v24_0_3_release(cleanup_manifest)
    patch_release_review = (_authenticate_successor_once(
                            patch_manifest, 'v24_0_2_release_review', reviewed_v24_0_2_release_contract)
                            if 'v24_0_2_release_review' in manifest else None)
    semantic_manifest = _without_reviewed_v24_0_2_release(patch_manifest)
    semantic_release_review = (_authenticate_successor_once(
                               semantic_manifest, 'v24_release_review', reviewed_v24_release_contract)
                               if 'v24_release_review' in manifest else None)
    throughput_successors = ((throughput_release_review,) if throughput_release_review is not None else ())
    cleanup_successors = (((cleanup_release_review,) if cleanup_release_review is not None else ())
                          + throughput_successors)
    tta_successors = (((tta_release_review,) if tta_release_review is not None else ())
                      + cleanup_successors)
    patch_successors = (((patch_release_review,) if patch_release_review is not None else ())
                        + tta_successors)
    semantic_successors = (((semantic_release_review,) if semantic_release_review is not None else ())
                           + patch_successors)
    publication_successors = (((publication_release_review,) if publication_release_review is not None else ())
                              + semantic_successors)
    performance_successors = (((performance_release_review,) if performance_release_review is not None else ())
                              + publication_successors)
    reconciliation_successors = (((reconciliation_release_review,) if reconciliation_release_review is not None else ())
                                 + performance_successors)
    patches = (*patches, *reconciliation_successors)
    removed_modules = {item['module'] for review in patches for item in review.get('module_snapshots', ())
                       if item.get('removed') is True}
    patch_definitions = {
        (item['module'], item['name']): item for review in patches for item in review['definitions']
    }
    retired_definitions = {
        (item['module'], item['name']): item for review in patches
        for item in review.get('removed_definitions', ())
    }
    retired_statements = Counter((item['module'], item['previous_sha256']) for review in patches
                                 for item in review.get('removed_statements', ()))
    retired_statement_hashes = set(retired_statements)
    v21_definitions = {(item['module'], item['name']): item for item in v21['definitions']}
    v21_seams = {
        (item['module'], item['name']): (item['definition_sha256'], item['seam_sha256'])
        for item in v21['local_import_seams']
    }

    def patched_definition_hash(module: str, name: str, previous_hash: str) -> str:
        for review in patches:
            record = next((item for item in review['definitions']
                           if (item['module'], item['name']) == (module, name)), None)
            if record is not None:
                if record['previous_sha256'] != previous_hash:
                    raise RuntimeError(f'v{review["release"]} supersession does not match its historical pin: {module}.{name}')
                previous_hash = str(record['sha256'])
        return previous_hash

    def reviewed_definition_hash(module: str, name: str, previous_hash: str) -> str:
        record = v21_definitions.get((module, name))
        if record is None:
            return patched_definition_hash(module, name, previous_hash)
        if record['previous_sha256'] != previous_hash:
            raise RuntimeError(f'v21 supersession does not match its historical pin: {module}.{name}')
        return patched_definition_hash(module, name, str(record['sha256']))

    available: dict[str, Counter[str]] = {}
    trees: dict[str, ast.Module] = {}
    top_level: dict[str, list[ast.stmt]] = {}
    local_import_seams: dict[tuple[str, str], tuple[str, str]] = {}
    audited_modules = (
        {str(item["module"]) for item in manifest["statements"]}
        | {module for module, _name in REVIEWED_V20_ADDED_DEFINITIONS}
        | {module for module, _label in REVIEWED_V20_ADDED_STATEMENTS}
        | {item['module'] for item in v21['definitions']}
        | {item['module'] for item in v21['statements']}
        | {item['module'] for item in v21['preserved_radial_modules']}
        | {item['module'] for review in patches for item in review['definitions'] + review['statements']}
        | {item['module'] for item in augmentation_review['definition_relocations']}
        | {item['module'] for item in pta_throughput_review['module_snapshots']}
    )
    for module in audited_modules:
        module_path = PACKAGE / f"{module}.py"
        if module in removed_modules:
            if module_path.exists():
                raise RuntimeError(f'Reviewed removed module is still present: {module}')
            continue
        module_source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(module_source, filename=str(module_path))
        trees[module] = tree
        top_level[module] = list(tree.body)
        available[module] = Counter(digest(node) for node in tree.body)
        local_import_seams.update(reviewed_local_import_seams(module, module_source, tree))

    verify_augmentation_relocations(augmentation_review, top_level)

    for (module, name), (expected_hash, reason) in REVIEWED_V20_ADDED_DEFINITIONS.items():
        expected_hash = reviewed_definition_hash(module, name, expected_hash)
        if (module, name) in retired_definitions:
            if retired_definitions[module, name]['previous_sha256'] != expected_hash:
                raise RuntimeError(f'retired v20 definition predecessor changed: {module}.{name}')
            continue
        matches = [node for node in top_level.get(module, ()) if getattr(node, 'name', None) == name]
        if not reason or len(matches) != 1 or digest(matches[0]) != expected_hash:
            raise RuntimeError(f'v20 reviewed added definition changed or is missing: {module}.{name}')
    current_v20_statements = reviewed_v20_statement_hashes((v21, *patches))
    for (module, label), (_historical_hash, reason) in REVIEWED_V20_ADDED_STATEMENTS.items():
        expected_hash = current_v20_statements[(module, label)]
        if (module, expected_hash) in retired_statement_hashes:
            continue
        if not reason or available.get(module, Counter())[expected_hash] != 1:
            raise RuntimeError(f'v20 reviewed added statement changed or is missing: {module}.{label}')

    for (module, name), record in v21_definitions.items():
        if (module, name) in retired_definitions:
            if retired_definitions[module, name]['previous_sha256'] != patched_definition_hash(
                    module, name, record['sha256']):
                raise RuntimeError(f'retired v21 definition predecessor changed: {module}.{name}')
            continue
        matches = [node for node in top_level[module] if getattr(node, 'name', None) == name]
        if len(matches) != 1 or digest(matches[0]) != patched_definition_hash(module, name, record['sha256']):
            raise RuntimeError(f'v21 reviewed definition changed or is missing: {module}.{name}')
    for (module, name), record in patch_definitions.items():
        if module in removed_modules:
            continue
        if (module, name) in retired_definitions:
            if retired_definitions[module, name]['previous_sha256'] != record['sha256']:
                raise RuntimeError(f'retired patch definition predecessor changed: {module}.{name}')
            continue
        matches = [node for node in top_level[module] if getattr(node, 'name', None) == name]
        if len(matches) != 1 or digest(matches[0]) != record['sha256']:
            raise RuntimeError(f'v21 patch reviewed definition changed or is missing: {module}.{name}')
    verify_policy_window_runtime_scope(policy_window_review, top_level['pipeline'],
                                       (tilted_azimuthal_review, backend_release_review, *reconciliation_successors))
    effective_statements = list(v21['statements'])
    for review in patches:
        replaced_statements = {(item['module'], item['previous_sha256']) for item in review['statements']}
        effective_statements = [
            item for item in effective_statements if (item['module'], item['sha256']) not in replaced_statements
        ] + review['statements']
    for item in effective_statements[:]:
        key = (item['module'], item['sha256'])
        if retired_statements[key]:
            effective_statements.remove(item)
            retired_statements[key] -= 1
    for item in effective_statements:
        if item['module'] in removed_modules:
            continue
        if available[item['module']][item['sha256']] != 1:
            raise RuntimeError(f'v21 reviewed statement changed or is missing: {item["module"]}.{item["label"]}')
    effective_definitions = {key: item for key, item in {**v21_definitions, **patch_definitions}.items()
                             if key not in retired_definitions}
    complete_modules = set(v21['complete_modules'])
    complete_modules.update(module for review in patches for module in review.get('complete_modules', ()))
    for module in complete_modules - removed_modules:
        expected = Counter(item['sha256'] for item in list(effective_definitions.values()) + effective_statements if item['module'] == module)
        if available[module] != expected:
            raise RuntimeError(f'v21 complete-module statement coverage differs: {module}')
    verify_pta_source_snapshots(pta_throughput_review, trees, (backend_release_review, *reconciliation_successors))
    verify_v22_1_release_scope(version_release_review, top_level, (backend_release_review, *reconciliation_successors))
    verify_v22_2_source_snapshots(backend_release_review, trees, reconciliation_successors)
    if reconciliation_release_review is not None:
        verify_v22_3_source_snapshots(reconciliation_release_review, trees, performance_successors)
        verify_v22_3_validation_tools(reconciliation_release_review, performance_successors)
    if performance_release_review is not None:
        verify_v22_3_source_snapshots(performance_release_review, trees, publication_successors)
        verify_v22_3_validation_tools(performance_release_review, publication_successors)
    if publication_release_review is not None:
        verify_v22_3_source_snapshots(publication_release_review, trees, semantic_successors)
        verify_v22_3_validation_tools(publication_release_review, semantic_successors)
    if semantic_release_review is not None:
        verify_v22_3_source_snapshots(semantic_release_review, trees, patch_successors)
        verify_v22_3_validation_tools(semantic_release_review, patch_successors)
    if patch_release_review is not None:
        verify_v22_3_source_snapshots(patch_release_review, trees, tta_successors)
        verify_v22_3_validation_tools(patch_release_review, tta_successors)
    if tta_release_review is not None:
        verify_v22_3_source_snapshots(tta_release_review, trees, cleanup_successors)
        verify_v22_3_validation_tools(tta_release_review, cleanup_successors)
    if cleanup_release_review is not None:
        verify_v22_3_source_snapshots(cleanup_release_review, trees, throughput_successors)
        verify_v22_3_validation_tools(cleanup_release_review, throughput_successors)
    if throughput_release_review is not None:
        verify_v22_3_source_snapshots(throughput_release_review, trees)
        verify_v22_3_validation_tools(throughput_release_review)
    for module, expected_hash in reviewed_radial_module_hashes(v21, patches).items():
        source = (PACKAGE / f'{module}.py').read_text(encoding='utf-8')
        if hashlib.sha256(source.encode('utf-8')).hexdigest() != expected_hash:
            raise RuntimeError(f'v21 changed a preserved Radial module: {module}')
    radial_method_hashes = reviewed_radial_definition_hashes(v21, patches)
    for (module, qualified_name), expected_hash in radial_method_hashes.items():
        scope = trees[module]
        for name in qualified_name.split('.'):
            matches = [node for node in scope.body if getattr(node, 'name', None) == name]
            if len(matches) != 1:
                raise RuntimeError(f'v21 preserved Radial definition is missing: {qualified_name}')
            scope = matches[0]
        if digest(scope) != expected_hash:
            raise RuntimeError(f'v21 changed preserved Radial arithmetic: {qualified_name}')

    expected_local_import_seams = {**REVIEWED_LOCAL_IMPORT_SEAMS, **v21_seams}
    for review in patches:
        seen_seams = set()
        for item in review.get('local_import_seam_updates', ()):
            key = (item['module'], item['name'])
            if key in seen_seams or key not in expected_local_import_seams or not item.get('reason'):
                raise RuntimeError('unknown, duplicate or unexplained local-import seam update')
            seen_seams.add(key)
            previous = (item.get('previous_definition_sha256'), item.get('previous_seam_sha256'))
            if previous != expected_local_import_seams[key]:
                raise RuntimeError('local-import seam update does not match its predecessor')
            current = (item.get('definition_sha256'), item.get('seam_sha256'))
            if any(not isinstance(value, str) or len(value) != 64 or
                   any(char not in '0123456789abcdef' for char in value) for value in current):
                raise RuntimeError('local-import seam update has an invalid digest')
            expected_local_import_seams[key] = current
        for item in review.get('removed_definitions', ()):
            key = (item['module'], item['name'])
            if key in expected_local_import_seams:
                if expected_local_import_seams[key][0] != item['previous_sha256']:
                    raise RuntimeError(f'retired local-import seam predecessor changed: {key}')
                del expected_local_import_seams[key]
    for key, (previous_hash, _previous_seam) in REVIEWED_LOCAL_IMPORT_SEAMS.items():
        if key in v21_seams:
            reviewed_definition_hash(*key, previous_hash)

    unexpected_local_import_seams = sorted(
        set(local_import_seams) - set(expected_local_import_seams)
    )
    missing_local_import_seams = sorted(
        set(expected_local_import_seams) - set(local_import_seams)
    )
    changed_local_import_seams = sorted(
        key
        for key in set(local_import_seams) & set(expected_local_import_seams)
        if local_import_seams[key] != expected_local_import_seams[key]
    )
    if unexpected_local_import_seams or missing_local_import_seams or changed_local_import_seams:
        raise RuntimeError(
            "local-import seam review mismatch: "
            f"unexpected={unexpected_local_import_seams!r}, "
            f"missing={missing_local_import_seams!r}, "
            f"changed={changed_local_import_seams!r}"
        )

    effective_changed = (INTENTIONALLY_CHANGED | set(REVIEWED_LOCAL_IMPORT_SEAMS)) - set(retired_definitions)

    inventory_keys = {
        (str(item["module"]), str(item["sha256"]))
        for item in manifest["statements"]
    }
    untracked_changed_bindings = sorted(
        set(INTENTIONALLY_CHANGED_BINDINGS) - inventory_keys
    )
    if untracked_changed_bindings:
        raise RuntimeError(
            "reviewed changed-binding entries are absent from the immutable inventory: "
            f"{untracked_changed_bindings!r}"
        )
    untracked_v20 = sorted(set(REVIEWED_V20_STATEMENT_REPLACEMENTS) - inventory_keys)
    if untracked_v20:
        raise RuntimeError(f'v20 replacement reviews are absent from the immutable inventory: {untracked_v20!r}')
    for (module, _baseline_hash), (name, replacement_hash, reason) in REVIEWED_V20_STATEMENT_REPLACEMENTS.items():
        replacement_hash = reviewed_definition_hash(module, name, replacement_hash)
        matches = [node for node in top_level[module] if getattr(node, 'name', None) == name]
        if not reason or len(matches) != 1 or digest(matches[0]) != replacement_hash:
            raise RuntimeError(f'v20 reviewed definition changed or is missing: {module}.{name}')
    untracked_pruning = sorted(
        (set(INTENTIONALLY_REMOVED) | set(INTENTIONALLY_PRUNED_REPLACEMENTS))
        - inventory_keys
    )
    if untracked_pruning:
        raise RuntimeError(
            f"reviewed pruning entries are absent from the immutable inventory: {untracked_pruning!r}"
        )

    retired_names = {
        (module, name)
        for (module, _statement_hash), name in INTENTIONALLY_REMOVED.items()
    }
    retired_names.update(
        (module, name)
        for (module, _statement_hash), (_replacement_hash, name)
        in INTENTIONALLY_PRUNED_REPLACEMENTS.items()
    )
    remaining_retired_names: list[tuple[str, str]] = []
    for module, name in sorted(retired_names):
        for node in ast.walk(trees[module]):
            declares_name = (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == name
            ) or (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, (ast.Store, ast.Del))
                and node.id == name
            )
            if declares_name:
                remaining_retired_names.append((module, name))
                break
    if remaining_retired_names:
        raise RuntimeError(
            f"reviewed dead bindings still exist: {remaining_retired_names!r}"
        )

    missing_pruned_replacements = [
        (module, replacement_hash)
        for (module, _baseline_hash), (replacement_hash, _name)
        in INTENTIONALLY_PRUNED_REPLACEMENTS.items()
        if (module, replacement_hash) not in retired_statement_hashes
        and available[module][replacement_hash] != 1
    ]
    if missing_pruned_replacements:
        raise RuntimeError(
            "missing or duplicate reviewed pruning replacements: "
            f"{missing_pruned_replacements!r}"
        )

    missing_changed = [
        (module, name)
        for module, name in sorted(effective_changed)
        if sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == INTENTIONALLY_RENAMED_CHANGED.get((module, name), name)
            for node in top_level[module]
        ) != 1
    ]
    if missing_changed:
        raise RuntimeError(f"missing or duplicate reviewed seam functions: {missing_changed!r}")
    for (module, public_name), implementation_name in INTENTIONALLY_RENAMED_CHANGED.items():
        public_matches = sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == public_name
            for node in top_level[module]
        )
        if public_matches != 1:
            raise RuntimeError(
                f"missing or duplicate public wrapper for {module}.{implementation_name}: "
                f"{module}.{public_name}"
            )

    missing_versions: list[tuple[str, str]] = []
    for (module, _baseline_hash), variable_name in INTENTIONALLY_VERSIONED.items():
        matches = 0
        for node in top_level[module]:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            matches += sum(isinstance(target, ast.Name) and target.id == variable_name for target in targets)
        if matches != 1:
            missing_versions.append((module, variable_name))
    if missing_versions:
        raise RuntimeError(f"missing or duplicate version declarations: {missing_versions!r}")

    missing_changed_bindings: list[tuple[str, str]] = []
    for (module, _baseline_hash), variable_name in INTENTIONALLY_CHANGED_BINDINGS.items():
        matches = 0
        for node in top_level[module]:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            matches += sum(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in targets
            )
        if matches != 1:
            missing_changed_bindings.append((module, variable_name))
    if missing_changed_bindings:
        raise RuntimeError(
            "missing or duplicate reviewed changed bindings: "
            f"{missing_changed_bindings!r}"
        )

    missing: list[dict[str, object]] = []
    preserved = 0
    changed = 0
    removed = 0
    for item in manifest["statements"]:
        module = str(item["module"])
        name = current_baseline_name(item.get("name"))
        statement_hash = str(item["sha256"])
        inventory_key = (module, statement_hash)
        if inventory_key in INTENTIONALLY_REMOVED:
            removed += 1
            continue
        if inventory_key in INTENTIONALLY_PRUNED_REPLACEMENTS:
            replacement_hash, _removed_name = INTENTIONALLY_PRUNED_REPLACEMENTS[inventory_key]
            if (module, replacement_hash) in retired_statement_hashes:
                removed += 1
                continue
            available[module][replacement_hash] -= 1
            changed += 1
            continue
        if inventory_key in REVIEWED_V20_STATEMENT_REPLACEMENTS:
            _name, replacement_hash, _reason = REVIEWED_V20_STATEMENT_REPLACEMENTS[inventory_key]
            replacement_hash = reviewed_definition_hash(module, _name, replacement_hash)
            if (module, _name) in retired_definitions:
                if retired_definitions[module, _name]['previous_sha256'] != replacement_hash:
                    raise RuntimeError(f'retired baseline predecessor changed: {module}.{_name}')
                removed += 1
                continue
            available[module][replacement_hash] -= 1
            changed += 1
            continue
        destination = INTENTIONALLY_RELOCATED.get((module, statement_hash), module)
        expected_hash = rename_replacements.get(inventory_key, statement_hash)
        if (destination, name) in retired_definitions:
            if (destination, name) not in REVIEWED_LOCAL_IMPORT_SEAMS:
                retired_hash = reviewed_definition_hash(destination, name, expected_hash)
                if retired_definitions[destination, name]['previous_sha256'] != retired_hash:
                    raise RuntimeError(f'retired baseline predecessor changed: {destination}.{name}')
            # Reviewed seam pins authenticate their own definition lineage above.
            removed += 1
            continue
        if (
            (module, name) in effective_changed
            or (module, statement_hash) in INTENTIONALLY_VERSIONED
            or inventory_key in INTENTIONALLY_CHANGED_BINDINGS
        ):
            changed += 1
            continue
        if (destination, name) in patch_definitions:
            expected_hash = reviewed_definition_hash(destination, name, expected_hash)
            if available[destination][expected_hash] < 1:
                missing.append(item)
                continue
            available[destination][expected_hash] -= 1
            changed += 1
            continue
        statement_changed = False
        for review in (backend_release_review, *reconciliation_successors):
            current_statement = next((record for record in review['statements']
                                      if (record['module'], record['previous_sha256']) == (destination, expected_hash)), None)
            if current_statement is not None:
                expected_hash = current_statement['sha256']
                statement_changed = True
        if (destination, expected_hash) in retired_statement_hashes:
            removed += 1
            continue
        if statement_changed:
            if available[destination][expected_hash] < 1:
                missing.append(item)
                continue
            available[destination][expected_hash] -= 1
            changed += 1
            continue
        if available[destination][expected_hash] < 1:
            missing.append(item)
            continue
        available[destination][expected_hash] -= 1
        preserved += 1

    if missing:
        raise RuntimeError(f"missing {len(missing)} preserved statements: {missing!r}")
    expected = int(manifest["statement_count"])
    if preserved + changed + removed != expected:
        raise RuntimeError(
            "statement accounting mismatch: "
            f"{preserved} preserved + {changed} changed + {removed} removed != {expected}"
        )
    print(
        "package inventory verified: "
        f"preserved={preserved}, reviewed_changes={changed}, reviewed_removals={removed}"
    )


def main() -> None:
    """Reuse successful checks only within this independent verification pass."""
    digest_token = _ACTIVE_DIGEST_CACHE.set({})
    review_token = _ACTIVE_REVIEW_AUTH_CACHE.set(set())
    try:
        _verify_main()
    finally:
        _ACTIVE_REVIEW_AUTH_CACHE.reset(review_token)
        _ACTIVE_DIGEST_CACHE.reset(digest_token)


if __name__ == "__main__":
    main()
