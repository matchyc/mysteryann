#include <gtest/gtest.h>
#include <omp.h>

#include <algorithm>
#include <boost/dynamic_bitset.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "mysteryann/distance.h"
#include "mysteryann/neighbor.h"
#include "mysteryann/parameters.h"
#include "mysteryann/util.h"
#include "index_bipartite.h"

namespace po = boost::program_options;
// std::vector<uint32_t> test_connected_q = {
//     0,    2,    3,    4,    5,    6,    7,    8,    10,   11,   12,   13,   15,   16,   17,   18,   19,   20,   23,
//     24,   25,   26,   27,   28,   29,   30,   31,   32,   33,   35,   36,   37,   38,   40,   41,   42,   43,   44,
//     45,   46,   47,   48,   49,   50,   51,   52,   53,   54,   55,   56,   57,   58,   60,   61,   62,   63,   64,
//     65,   66,   68,   70,   71,   72,   73,   75,   77,   78,   80,   81,   82,   83,   84,   85,   86,   87,   88,
//     89,   90,   91,   92,   93,   94,   95,   96,   97,   98,   99,   100,  101,  104,  106,  107,  108,  109,  110,
//     111,  113,  114,  115,  117,  118,  120,  121,  122,  123,  124,  125,  126,  127,  128,  129,  130,  132,  133,
//     134,  135,  136,  137,  139,  140,  141,  142,  143,  144,  145,  146,  148,  149,  150,  151,  152,  153,  154,
//     156,  157,  158,  159,  160,  161,  162,  163,  164,  166,  167,  168,  169,  170,  171,  172,  173,  175,  176,
//     177,  179,  180,  181,  182,  183,  184,  185,  186,  188,  189,  190,  191,  192,  193,  194,  195,  196,  198,
//     199,  201,  202,  203,  204,  205,  206,  207,  208,  209,  210,  211,  212,  213,  215,  216,  217,  218,  219,
//     220,  221,  222,  223,  224,  226,  227,  228,  229,  230,  231,  232,  233,  234,  235,  236,  237,  239,  240,
//     242,  243,  244,  245,  246,  247,  248,  249,  250,  253,  254,  255,  256,  257,  258,  259,  260,  261,  262,
//     263,  265,  266,  267,  268,  269,  270,  272,  273,  274,  275,  276,  277,  278,  279,  280,  281,  282,  283,
//     284,  285,  286,  287,  289,  291,  292,  293,  294,  295,  296,  297,  298,  299,  300,  301,  302,  303,  304,
//     305,  306,  308,  309,  311,  312,  313,  314,  315,  316,  318,  319,  320,  321,  323,  324,  325,  326,  327,
//     328,  329,  330,  331,  332,  333,  334,  335,  336,  338,  339,  340,  341,  342,  343,  344,  345,  346,  347,
//     348,  349,  350,  351,  352,  353,  354,  355,  356,  357,  358,  359,  360,  362,  363,  364,  365,  366,  367,
//     368,  369,  370,  372,  373,  374,  375,  376,  377,  378,  379,  380,  381,  382,  384,  385,  386,  387,  388,
//     389,  390,  391,  392,  393,  394,  395,  396,  397,  398,  399,  400,  403,  404,  405,  406,  407,  408,  409,
//     410,  412,  413,  414,  415,  416,  417,  419,  420,  421,  422,  423,  424,  425,  426,  427,  428,  429,  430,
//     431,  432,  433,  434,  435,  436,  437,  438,  439,  440,  441,  442,  443,  444,  445,  447,  448,  449,  450,
//     451,  453,  454,  455,  456,  457,  458,  460,  461,  462,  463,  465,  467,  468,  469,  470,  471,  472,  473,
//     474,  475,  476,  477,  478,  479,  480,  481,  482,  483,  484,  485,  486,  488,  489,  490,  491,  492,  493,
//     494,  495,  496,  497,  498,  500,  501,  503,  505,  506,  509,  510,  511,  512,  513,  514,  515,  516,  517,
//     518,  519,  520,  521,  522,  523,  524,  525,  526,  527,  528,  529,  530,  532,  533,  534,  535,  536,  537,
//     538,  540,  541,  542,  543,  544,  545,  547,  548,  549,  550,  551,  552,  553,  554,  555,  557,  559,  560,
//     561,  562,  563,  564,  566,  567,  569,  570,  571,  572,  573,  574,  575,  576,  577,  579,  580,  581,  582,
//     583,  584,  585,  586,  587,  588,  589,  590,  592,  593,  594,  595,  596,  597,  598,  599,  600,  601,  602,
//     603,  604,  605,  606,  607,  609,  610,  611,  612,  615,  616,  617,  620,  621,  622,  623,  624,  625,  626,
//     628,  629,  630,  631,  632,  633,  635,  636,  637,  638,  639,  641,  642,  643,  645,  646,  647,  648,  649,
//     650,  652,  653,  654,  655,  656,  657,  658,  659,  660,  661,  662,  663,  664,  665,  666,  667,  668,  669,
//     671,  673,  674,  675,  676,  677,  679,  680,  682,  683,  684,  685,  686,  687,  688,  689,  690,  691,  692,
//     693,  694,  695,  696,  697,  698,  699,  700,  701,  702,  703,  704,  705,  706,  707,  708,  709,  710,  711,
//     712,  713,  714,  715,  717,  718,  719,  720,  721,  722,  723,  725,  726,  727,  728,  730,  731,  732,  733,
//     734,  735,  736,  737,  739,  740,  741,  742,  743,  744,  745,  747,  749,  750,  751,  752,  753,  754,  756,
//     759,  760,  761,  762,  763,  764,  765,  766,  767,  768,  769,  770,  771,  772,  773,  774,  775,  776,  777,
//     778,  779,  780,  781,  782,  783,  784,  785,  786,  789,  791,  792,  793,  794,  795,  796,  797,  798,  799,
//     800,  801,  802,  803,  804,  805,  806,  807,  808,  809,  810,  811,  812,  813,  814,  815,  816,  817,  818,
//     819,  820,  821,  822,  823,  824,  825,  826,  827,  828,  829,  831,  832,  833,  834,  835,  836,  837,  838,
//     839,  840,  841,  842,  843,  844,  845,  846,  848,  849,  850,  851,  852,  853,  854,  855,  856,  857,  861,
//     862,  864,  865,  866,  867,  868,  869,  870,  871,  872,  873,  874,  875,  876,  877,  878,  879,  880,  881,
//     882,  883,  884,  885,  886,  887,  888,  890,  891,  892,  893,  894,  896,  897,  898,  899,  900,  901,  902,
//     903,  904,  906,  907,  908,  909,  911,  912,  913,  914,  915,  916,  917,  918,  919,  920,  921,  922,  923,
//     924,  925,  927,  928,  929,  930,  931,  932,  933,  934,  935,  936,  937,  939,  940,  941,  942,  943,  944,
//     945,  947,  948,  949,  950,  951,  952,  954,  955,  956,  957,  958,  960,  961,  962,  963,  964,  966,  967,
//     968,  969,  970,  971,  973,  974,  975,  976,  977,  978,  979,  980,  981,  982,  983,  984,  985,  986,  987,
//     988,  992,  993,  995,  996,  997,  999,  1000, 1001, 1002, 1004, 1005, 1006, 1007, 1008, 1010, 1011, 1012, 1013,
//     1014, 1015, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1033, 1035,
//     1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054,
//     1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1067, 1068, 1069, 1070, 1072, 1073, 1074, 1076,
//     1077, 1078, 1079, 1080, 1081, 1082, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095,
//     1096, 1097, 1098, 1100, 1101, 1103, 1104, 1105, 1106, 1107, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117,
//     1118, 1120, 1121, 1122, 1123, 1124, 1126, 1128, 1129, 1130, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140,
//     1141, 1142, 1143, 1144, 1145, 1147, 1148, 1149, 1151, 1152, 1154, 1155, 1156, 1158, 1159, 1160, 1161, 1162, 1163,
//     1164, 1165, 1166, 1167, 1168, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1182, 1183, 1184,
//     1185, 1186, 1187, 1188, 1190, 1192, 1193, 1194, 1196, 1197, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207,
//     1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1220, 1221, 1222, 1224, 1225, 1227, 1228, 1229,
//     1230, 1231, 1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240, 1241, 1243, 1244, 1245, 1246, 1247, 1248, 1249, 1250,
//     1251, 1252, 1253, 1254, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1265, 1266, 1267, 1268, 1269, 1270,
//     1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1281, 1282, 1283, 1284, 1285, 1286, 1287, 1288, 1290, 1291,
//     1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1302, 1303, 1304, 1305, 1306, 1307, 1309, 1310, 1311, 1312,
//     1313, 1314, 1315, 1317, 1318, 1319, 1320, 1321, 1323, 1325, 1326, 1327, 1329, 1330, 1331, 1332, 1334, 1336, 1337,
//     1338, 1339, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1352, 1353, 1354, 1355, 1356, 1357, 1358,
//     1359, 1360, 1362, 1363, 1364, 1365, 1367, 1368, 1369, 1370, 1371, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380,
//     1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399,
//     1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1415, 1417, 1418, 1419,
//     1420, 1421, 1422, 1423, 1424, 1425, 1426, 1427, 1428, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439,
//     1440, 1441, 1442, 1443, 1445, 1446, 1447, 1449, 1450, 1451, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1460, 1461,
//     1462, 1463, 1464, 1465, 1467, 1468, 1469, 1470, 1471, 1472, 1473, 1474, 1475, 1477, 1478, 1479, 1481, 1482, 1483,
//     1484, 1485, 1486, 1487, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1498, 1499, 1500, 1501, 1503, 1504,
//     1506, 1507, 1508, 1509, 1510, 1511, 1512, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1522, 1523, 1524,
//     1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1535, 1541, 1542, 1543, 1544, 1545, 1546, 1547, 1548,
//     1549, 1550, 1551, 1553, 1554, 1555, 1556, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568,
//     1569, 1570, 1571, 1572, 1573, 1574, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588,
//     1589, 1591, 1592, 1593, 1595, 1596, 1597, 1598, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1608, 1609, 1610, 1611,
//     1612, 1613, 1616, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1628, 1629, 1630, 1631, 1633, 1634, 1635,
//     1636, 1637, 1638, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1652, 1653, 1654, 1655, 1656, 1657,
//     1658, 1659, 1660, 1662, 1663, 1664, 1665, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1678, 1679, 1680,
//     1682, 1683, 1684, 1685, 1686, 1688, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1700, 1701, 1702,
//     1703, 1704, 1706, 1707, 1708, 1709, 1711, 1712, 1713, 1714, 1715, 1716, 1718, 1719, 1720, 1721, 1722, 1724, 1727,
//     1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739, 1740, 1741, 1742, 1743, 1744, 1745, 1746,
//     1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756, 1757, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1767,
//     1768, 1769, 1770, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1780, 1781, 1782, 1783, 1784, 1786, 1787, 1788,
//     1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807,
//     1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820, 1821, 1823, 1824, 1825, 1827, 1828, 1829,
//     1831, 1832, 1833, 1835, 1836, 1838, 1839, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852,
//     1853, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1866, 1867, 1869, 1870, 1871, 1873, 1874, 1875,
//     1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1888, 1890, 1891, 1892, 1893, 1894, 1895, 1896,
//     1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915,
//     1916, 1917, 1918, 1920, 1921, 1922, 1924, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938,
//     1939, 1940, 1942, 1943, 1944, 1945, 1946, 1947, 1949, 1950, 1951, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960,
//     1961, 1962, 1963, 1964, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981,
//     1982, 1983, 1984, 1985, 1986, 1988, 1989, 1990, 1991, 1993, 1994, 1996, 1997, 1998, 1999,

// };
float ComputeRecall(uint32_t q_num, uint32_t k, uint32_t gt_dim, uint32_t *res, uint32_t *gt) {
    uint32_t total_count = 0;
    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<uint32_t> one_gt(gt + i * gt_dim, gt + i * gt_dim + k);
        std::vector<uint32_t> intersection;
        std::vector<uint32_t> temp_res(res + i * k, res + i * k + k);
        // check if there duplication in temp_res
        // std::sort(temp_res.begin(), temp_res.end());

        // std::sort(one_gt.begin(), one_gt.end());
        for (auto p : one_gt) {
            if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end()) intersection.push_back(p);
        }
        // std::set_intersection(temp_res.begin(), temp_res.end(), one_gt.begin(), one_gt.end(),
        //   std::back_inserter(intersection));

        total_count += static_cast<uint32_t>(intersection.size());
    }
    return static_cast<float>(total_count) / (float)(k * q_num);
    // return static_cast<float>(total_count) / (k * test_connected_q.size());
}

double ComputeRderr(float* gt_dist, uint32_t gt_dim, std::vector<std::vector<float>>& res_dists, uint32_t k, mysteryann::Metric metric) {
    double total_err = 0;
    uint32_t q_num = res_dists.size();

    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<float> one_gt(gt_dist + i * gt_dim, gt_dist + i * gt_dim + k);
        std::vector<float> temp_res(res_dists[i].begin(), res_dists[i].end());
        if (metric == mysteryann::INNER_PRODUCT) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = -1.0 * temp_res[j];
            }
        } else if (metric == mysteryann::COSINE) {
            for (size_t j = 0; j < k; ++j) {
                temp_res[j] = 2.0 * ( 1.0 - (-1.0 * temp_res[j]));
            }
        }
        double err = 0.0;
        for (uint32_t j = 0; j < k; j++) {
            err += std::fabs(temp_res[j] - one_gt[j]) / double(one_gt[j]);
        }
        err = err / static_cast<double>(k);
        total_err = total_err + err;
    }
    return total_err / static_cast<double>(q_num);
}

int main(int argc, char **argv) {
    std::string base_data_file;
    std::string query_file;
    std::string sampled_query_data_file;
    std::string gt_file;

    std::string bipartite_index_save_file, projection_index_save_file;
    std::string data_type;
    std::string dist;
    std::vector<uint32_t> L_vec;
    // uint32_t L_pq;
    uint32_t num_threads;
    uint32_t k;
    std::string evaluation_save_path;

    po::options_description desc{"Arguments"};
    try {
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("data_type", po::value<std::string>(&data_type)->required(), "data type <int8/uint8/float>");
        desc.add_options()("dist", po::value<std::string>(&dist)->required(), "distance function <l2/ip>");
        desc.add_options()("base_data_path", po::value<std::string>(&base_data_file)->required(),
                           "Input data file in bin format");
        desc.add_options()("sampled_query_data_path", po::value<std::string>(&sampled_query_data_file)->required(),
                           "Sampled query file in bin format");
        desc.add_options()("query_path", po::value<std::string>(&query_file)->required(), "Query file in bin format");
        desc.add_options()("gt_path", po::value<std::string>(&gt_file)->required(), "Groundtruth file in bin format");
        // desc.add_options()("query_data_path",
        //                    po::value<std::string>(&query_data_file)->required(),
        //                    "Query file in bin format");
        desc.add_options()("bipartite_index_save_path", po::value<std::string>(&bipartite_index_save_file)->required(),
                           "Path prefix for saving bipartite index file components");
        desc.add_options()("projection_index_save_path",
                           po::value<std::string>(&projection_index_save_file)->required(),
                           "Path prefix for saving projetion index file components");
        desc.add_options()("L_pq", po::value<std::vector<uint32_t>>(&L_vec)->multitoken()->required(),
                           "Priority queue length for searching");
        desc.add_options()("k", po::value<uint32_t>(&k)->default_value(1)->required(), "k nearest neighbors");
        desc.add_options()("evaluation_save_path", po::value<std::string>(&evaluation_save_path),
                           "Path prefix for saving evaluation results");
        // desc.add_options()(
        //     "alpha", po::value<float>(&alpha)->default_value(1.2f),
        //     "alpha controls density and diameter of graph, set 1 for sparse graph, "
        //     "1.2 or 1.4 for denser graphs with lower diameter");
        desc.add_options()("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                           "Number of threads used for building index (defaults to "
                           "omp_get_num_procs())");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << '\n';
        return -1;
    }
    uint32_t base_num, base_dim, sq_num, sq_dim;
    mysteryann::load_meta<float>(base_data_file.c_str(), base_num, base_dim);
    mysteryann::load_meta<float>(sampled_query_data_file.c_str(), sq_num, sq_dim);
    // mysteryann::IndexBipartite index_bipartite(base_dim, base_num + sq_num, mysteryann::INNER_PRODUCT, nullptr);

    // float *data_bp = nullptr;
    // float *data_sq = nullptr;
    // float *aligned_data_bp = nullptr;
    // float *aligned_data_sq = nullptr;
    mysteryann::Parameters parameters;
    // mysteryann::load_data<float>(base_data_file.c_str(), base_num, base_dim, data_bp);
    // mysteryann::load_data<float>(sampled_query_data_file.c_str(), sq_num, sq_dim, data_sq);
    // aligned_data_bp = mysteryann::data_align(data_bp, base_num, base_dim);
    // aligned_data_sq = mysteryann::data_align(data_sq, sq_num, sq_dim);

    parameters.Set<uint32_t>("num_threads", num_threads);
    omp_set_num_threads(num_threads);
    uint32_t q_pts, q_dim;
    mysteryann::load_meta<float>(query_file.c_str(), q_pts, q_dim);
    float *query_data = nullptr;
    mysteryann::load_data_search<float>(query_file.c_str(), q_pts, q_dim, query_data);
    // float *aligned_query_data = mysteryann::data_align(query_data, q_pts, q_dim);
    float *aligned_query_data = query_data;

    uint32_t gt_pts, gt_dim;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    mysteryann::load_gt_meta<uint32_t>(gt_file.c_str(), gt_pts, gt_dim);
    // mysteryann::load_gt_data<uint32_t>(gt_file.c_str(), gt_pts, gt_dim, gt_ids);
    mysteryann::load_gt_data_with_dist<uint32_t, float>(gt_file.c_str(), gt_pts, gt_dim, gt_ids, gt_dists);
    mysteryann::Metric dist_metric = mysteryann::INNER_PRODUCT;
    if (dist == "l2") {
        dist_metric = mysteryann::L2;
        std::cout << "Using l2 as distance metric" << std::endl;
    } else if (dist == "ip") {
        dist_metric = mysteryann::INNER_PRODUCT;
        std::cout << "Using inner product as distance metric" << std::endl;
    } else if (dist == "cosine") {
        dist_metric = mysteryann::COSINE;
        std::cout << "Using cosine as distance metric" << std::endl;
    } else {
        std::cout << "Unknown distance type: " << dist << std::endl;
        return -1;
    }
    mysteryann::IndexBipartite index(q_dim, base_num + sq_num, dist_metric, nullptr);

    index.LoadSearchNeededData(base_data_file.c_str(), sampled_query_data_file.c_str());

    std::cout << "Load graph index: " << projection_index_save_file << std::endl;
    index.LoadProjectionGraph(projection_index_save_file.c_str());
    // index.LoadNsgGraph(projection_index_save_file.c_str());
    if (index.need_normalize) {
        std::cout << "Normalizing query data" << std::endl;
        for (uint32_t i = 0; i < q_pts; i++) {
            mysteryann::normalize<float>(aligned_query_data + i * q_dim, q_dim);
        }
    }
    index.InitVisitedListPool(num_threads + 5);

    index.ConvertAdjList2CSR(index.row_ptr_, index.col_idx_);
    // index.Load(bipartite_index_save_file.c_str());
    // Search
    // uint32_t k = 1;
    std::cout << "k: " << k << std::endl;
    uint32_t *res = new uint32_t[q_pts * k];
    memset(res, 0, sizeof(uint32_t) * q_pts * k);
    std::vector<std::vector<float>> res_dists(q_pts, std::vector<float>(k, 0.0));
    uint32_t *projection_cmps_vec = (uint32_t *)aligned_alloc(4, sizeof(uint32_t) * q_pts);
    memset(projection_cmps_vec, 0, sizeof(uint32_t) * q_pts);
    float *projection_latency_vec = (float *)aligned_alloc(4, sizeof(float) * q_pts);
    memset(projection_latency_vec, 0, sizeof(float) * q_pts);
    std::ofstream evaluation_out;
    if (!evaluation_save_path.empty()) {
        evaluation_out.open(evaluation_save_path, std::ios::out);
    }
    for (uint32_t L_pq : L_vec) {
        if (k > L_pq) {
            std::cout << "L_pq must greater or equal than k" << std::endl;
            exit(1);
        }
        parameters.Set<uint32_t>("L_pq", L_pq);
        // record the search time
        auto start = std::chrono::high_resolution_clock::now();
// index.dist_cmp_metric.reset();
// index.memory_access_metric.reset();
// std::cout << "begin search" << std::endl;
#pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < q_pts; ++i) {
            // for (size_t i = 0; i < test_connected_q.size(); ++i) {
            // if (index.need_normalize) {
            //     // std::cout << "Normalizing query data" << std::endl;
            //     // for (uint32_t i = 0; i < q_pts; i++) {
            //     mysteryann::normalize<float>(aligned_query_data + i * q_dim, q_dim);
            //     // }
            // }   
            // auto q_start = std::chrono::high_resolution_clock::now();
            projection_cmps_vec[i] = index.SearchProjectionCSR(aligned_query_data + i * q_dim, k, i, parameters, res + i * k, res_dists[i]);
            // auto q_end = std::chrono::high_resolution_clock::now();
            // projection_latency_vec[i] = std::chrono::duration_cast<std::chrono::microseconds>(q_end - q_start).count();
            // auto q_start = std::chrono::high_resolution_clock::now();
            // size_t qid = test_connected_q[i];
            // projection_cmps_vec[qid] =
            //     index.SearchProjectionGraph(aligned_query_data + qid * q_dim, k, qid, parameters, res);
            // auto q_end = std::chrono::high_resolution_clock::now();
            // projection_latency_vec[qid] =
            //     std::chrono::duration_cast<std::chrono::microseconds>(q_end - q_start).count();
            // EXPECT_EQ(indices.size(), k);
            // for (auto &idx : indices) {
            //     EXPECT_LT(idx, nd_);
            // }
            // res.push_back(indices);
        }
        auto end = std::chrono::high_resolution_clock::now();

        // // index.dist_cmp_metric.print(std::string("distance: "));
        // // index.memory_access_metric.print(std::string("memory: "));
        // // index.block_metric.print(std::string("block: "));
        // std::cout << "Projection Search time: "
        //           << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" <<
        //           std::endl;
        // average latency
        // std::cout << "Projection Average latency: "
        //           << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / q_pts << "ms "
        //           << std::endl;
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        float qps = (float)q_pts / ((float)diff / 1000.0);
        float recall = ComputeRecall(q_pts, k, gt_dim, res, gt_ids);
        // std::cout << "Projection QPS: " << qps << " Qeury / Second" << std::endl;
        // std::cout << "Projection Testing Recall: " << recall << std::endl;
        float avg_projection_cmps = 0.0;
        for (size_t i = 0; i < q_pts; ++i) {
            avg_projection_cmps += projection_cmps_vec[i];
        }
        avg_projection_cmps /= q_pts;
        // avg_projection_cmps /= (float)test_connected_q.size();
        // std::cout << "Projection Search Average Cmps: " << avg_projection_cmps << std::endl;
        float avg_projection_latency = 0.0;
        for (size_t i = 0; i < q_pts; ++i) {
            avg_projection_latency += projection_latency_vec[i];
        }
        avg_projection_latency /= (float)q_pts;
        double rderr = ComputeRderr(gt_dists, gt_dim, res_dists, k, dist_metric);
        // avg_projection_latency /= (float)test_connected_q.size();
        // std::cout << "Projection Search Average Latency: " << avg_projection_latency << std::endl;
        std::cout << L_pq << "\t\t" << qps << "\t\t" << avg_projection_cmps << "\t\t"
                  << ((float)diff / q_pts) << "\t\t" << recall << "\t\t" << rderr << std::endl;
        // std::cout << "Directly divide latency: "
        //           << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) /
        //           (float)q_pts
        //           << std::endl;
        if (evaluation_out.is_open()) {
            evaluation_out << L_pq << "," << qps << "," << avg_projection_cmps << "," << ((float)diff / q_pts) << ","
                           << recall << "," << rderr << std::endl;
        }
    }
    if (evaluation_out.is_open()) {
        evaluation_out.close();
    }

    // uint32_t *cmps_vec = new uint32_t[q_pts];
    // float *latency_vec = new float[q_pts];
    // for (uint32_t L_pq : L_vec) {
    //     parameters.Set<uint32_t>("L_pq", L_pq);
    //     auto start = std::chrono::high_resolution_clock::now();

    //     // #pragma omp parallel for schedule(dynamic, 1)
    //     for (size_t i = 0; i < q_pts; ++i) {
    //         auto q_start = std::chrono::high_resolution_clock::now();
    //         cmps_vec[i] = index.SearchBipartiteGraph(aligned_query_data + i * q_dim, k, i, parameters, res);
    //         // EXPECT_EQ(indices.size(), k);
    //         // for (auto &idx : indices) {
    //         //     EXPECT_LT(idx, nd_);
    //         // }
    //         auto q_end = std::chrono::high_resolution_clock::now();
    //         float q_time = std::chrono::duration_cast<std::chrono::microseconds>(q_end - q_start).count();
    //         latency_vec[i] = q_time;
    //     }
    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::cout << "Bipartite Search time: "
    //               << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" <<
    //               std::endl;
    //     // average latency
    //     // std::cout << "Bipartite Average latency: "
    //     //           << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / q_pts << " ms"
    //     //           << std::endl;
    //     std::cout << "Bipartite QPS: "
    //               << (float)q_pts /
    //                      ((float)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0)
    //               << " Qeury / Second" << std::endl;
    //     std::cout << "Bipartite Testing Recall: " << ComputeRecall(q_pts, k, gt_dim, res, gt_ids) << std::endl;

    //     float avg_cmps = 0;
    //     for (size_t i = 0; i < q_pts; ++i) {
    //         avg_cmps += cmps_vec[i];
    //     }
    //     avg_cmps /= q_pts;
    //     std::cout << "Average comparisons: " << avg_cmps << std::endl;

    //     float avg_latency = 0;
    //     for (size_t i = 0; i < q_pts; ++i) {
    //         avg_latency += latency_vec[i];
    //     }
    //     avg_latency /= q_pts;
    //     std::cout << "Average latency: " << avg_latency << std::endl;
    // }
    // delete[] cmps_vec;
    // delete[] latency_vec;

    delete[] res;
    delete[] projection_cmps_vec;
    delete[] projection_latency_vec;
    delete[] aligned_query_data;
    delete[] gt_ids;

    return 0;
}