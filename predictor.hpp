#include <math.h>
#include <string.h>

#include <deque>
#include <iostream>
#include <map>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Pair_Trading.hpp"

#define UNUSED(x) ((void)(x))
struct Quotes {
    std::string symbol;
    float size;
    float price;
    std::string side;
};
class SpreadQuotes {
    bool ready;

   public:
    Quotes s;

   public:
    SpreadQuotes() { ready = false; }
    void set_size(float size);
    void set_price(float price);
    void set_side(std::string side);
    void set_symbol(std::string symbol);
    float get_size();
    float get_price();
    std::string get_side();
    std::string get_symbol();
};
float SpreadQuotes::get_size() {
    return s.size;
}
float SpreadQuotes::get_price() {
    return s.price;
}
std::string SpreadQuotes::get_side() {
    return s.side;
}
std::string SpreadQuotes::get_symbol() {
    return s.symbol;
}
void SpreadQuotes::set_size(float size) {
    s.size = size;
}
void SpreadQuotes::set_price(float price) {
    s.price = price;
}
void SpreadQuotes::set_side(std::string side) {
    s.side = side;
}
void SpreadQuotes::set_symbol(std::string Symbol) {
    s.symbol = Symbol;
}

class Spread {
    unsigned long window_size;
    bool is_warmed_up;
    std::deque<float> xs;

   public:
    Spread(unsigned long n = 120) : window_size(n), is_warmed_up(false) {}
    void update(float price);
    bool warmed_up();
};
bool Spread::warmed_up() {
    return is_warmed_up;
}
void Spread::update(float x) {
    if (xs.size() >= window_size) {
        xs.pop_front();
        xs.push_back(x);
    } else {
        xs.push_back(x);
        is_warmed_up = true;
    }
}
struct Table {
    float w1, w2, mu, std, model, capital;
};
struct Orderbook {
    float Buyprice[3];    //= {29999.5, 30000.5, 30001.5};
    float Sellprice[3];   // = {29999.0, 29990, 23293};
    float Buyvolume[3];   // = {1, 2, 3};
    float Sellvolume[3];  // = {2.1, 3.1, 4};
    float TimeStamp;      // = 161323023023;
};
class Predictor {
    Spread ref_spread;
    Spread target_spread;
    Table table;
    float ref_timestamp;
    float target_timestamp;
    int position;
    bool cointegration_check;
    bool timestamp_check;
    float ref_size;
    float target_size;
    float r_sec_timestamp;
    float t_sec_timestamp;
    std::string ref_symbol;
    std::string target_symbol;
    float slippage;

   public:
    SpreadQuotes SQ_R;
    SpreadQuotes SQ_T;

   public:
    Predictor(std::string r, std::string t, float spg) : ref_spread(), target_spread(), table(), ref_timestamp(0), target_timestamp(0), position(0), cointegration_check(false), timestamp_check(false), ref_size(0), target_size(0), r_sec_timestamp(0), t_sec_timestamp(0) {
        ref_symbol = r;
        target_symbol = t;
        slippage = spg;
    }

    void SetTable();
    std::tuple<float, float> get_asks(Orderbook *orderbook_r, Orderbook *orderbook_t);
    std::tuple<float, float> get_bids(Orderbook *orderbook_r, Orderbook *orderbook_t);
    std::tuple<float, float> get_level_asks(Orderbook *orderbook_r, Orderbook *orderbook_t);
    std::tuple<float, float> get_level_bids(Orderbook *orderbook_r, Orderbook *orderbook_t);
    void update_spreads(Orderbook *orderbook_r, Orderbook *orderbook_t);
    float slippage_number(float x, float size);
    std::string side_determination(float size);
    Table cointegration_test();
    void open_Quotes_setting(float ref_trade_price, float target_trade_price);
    void close_Quotes_setting(float ref_trade_price, float target_trade_price);
    std::tuple<SpreadQuotes, SpreadQuotes> get_predictor_spread_price(Orderbook *orderbook_r, Orderbook *orderbook_t, float opne_threshold, float stop_loss_threshold);
    // std::tuple<float, float, float, float, int, float> cointegration_test();
};
void Predictor::SetTable() {
    table.w1 = 0;
    table.w2 = 0;
    table.mu = 0;
    table.std = 0;
    table.model = 0;
    table.capital = 2000;
}
std::tuple<float, float> Predictor::get_asks(Orderbook *orderbook_r, Orderbook *orderbook_t) {
    return {orderbook_r->Sellprice[0], orderbook_t->Sellprice[0]};
}

std::tuple<float, float> Predictor::get_bids(Orderbook *orderbook_r, Orderbook *orderbook_t) {
    return {orderbook_r->Buyprice[0], orderbook_t->Buyprice[0]};
}

std::tuple<float, float> Predictor::get_level_asks(Orderbook *orderbook_r, Orderbook *orderbook_t) {
    return {(orderbook_r->Sellprice[0] + orderbook_r->Sellprice[1] + orderbook_r->Sellprice[2]) / 3,
            (orderbook_t->Sellprice[0] + orderbook_t->Sellprice[1] + orderbook_t->Sellprice[2]) / 3};
}
std::tuple<float, float> Predictor::get_level_bids(Orderbook *orderbook_r, Orderbook *orderbook_t) {
    return {(orderbook_r->Buyprice[0] + orderbook_r->Buyprice[1] + orderbook_r->Buyprice[2]) / 3,
            (orderbook_t->Buyprice[0] + orderbook_t->Buyprice[1] + orderbook_t->Buyprice[2]) / 3};
}
void Predictor::update_spreads(Orderbook *orderbook_r, Orderbook *orderbook_t) {
    if (orderbook_r && orderbook_t && orderbook_r->TimeStamp != ref_timestamp && orderbook_t->TimeStamp != target_timestamp) {
        target_timestamp = orderbook_t->TimeStamp;
        ref_timestamp = orderbook_r->TimeStamp;
        auto [ref_ask, target_ask] = get_asks(orderbook_r, orderbook_t);
        auto [ref_bid, target_bid] = get_bids(orderbook_r, orderbook_t);
        float ref_mid_price = (ref_ask + ref_bid) / 2;
        float target_mid_price = (target_ask + target_bid) / 2;
        ref_spread.update(ref_mid_price);
        target_spread.update(target_mid_price);
    }
}
void Predictor::open_Quotes_setting(float ref_trade_price, float target_trade_price) {
    float ref_size = table.w1 * table.capital / ref_trade_price;
    float target_size = table.w2 * table.capital / target_trade_price;
    SQ_R.set_symbol(ref_symbol);
    SQ_R.set_price(ref_trade_price * (1 + slippage_number(slippage, ref_size)));
    SQ_R.set_side(side_determination(ref_size));
    SQ_R.set_size(abs(ref_size));
    SQ_T.set_symbol(target_symbol);
    SQ_T.set_price(target_trade_price * (1 + slippage_number(slippage, target_size)));
    SQ_T.set_side(side_determination(target_size));
    SQ_T.set_size(abs(target_size));
}
void Predictor::close_Quotes_setting(float ref_trade_price, float target_trade_price) {
    float ref_size = table.w1 * table.capital / ref_trade_price;
    float target_size = table.w2 * table.capital / target_trade_price;
    SQ_R.set_symbol(ref_symbol);
    SQ_R.set_price(ref_trade_price * (1 + slippage_number(slippage, ref_size)));
    SQ_R.set_side(side_determination(ref_size));
    SQ_R.set_size(abs(ref_size));
    SQ_T.set_symbol(target_symbol);
    SQ_T.set_price(target_trade_price * (1 + slippage_number(slippage, target_size)));
    SQ_T.set_side(side_determination(target_size));
    SQ_T.set_size(abs(target_size));
}
Table Predictor::cointegration_test() {
    Table t;
    Stock_data data;
    vector<double> tmp;
    data = data.read_csv("./check_data/check_data_30.csv");
    data = data.to_log();
    // data.show_print();
    data.convert_to_mat();
    data.bic_tranformed();
    data.JCI_AutoSelection();
    data.Johanson_Mean();
    data.Johanson_Stdev();
    t.mu = data.Johanson_mean;
    t.std = data.Johanson_std;
    t.w1 = data.w1;
    t.w2 = data.w2;
    return t;
}

float Predictor::slippage_number(float x, float size) {
    float neg = x * (-1);
    if (position == -1) {
        return size > 0 ? neg : x;
    } else if (position == 1) {
        return size < 0 ? neg : x;
    }
}
std::string Predictor::side_determination(float size) {
    if (position == -1) {
        return size > 0 ? "SELL" : "BUY";
    } else if (position == 1) {
        return size < 0 ? "SELL" : "BUY";
    }
}
std::tuple<SpreadQuotes, SpreadQuotes> Predictor::get_predictor_spread_price(Orderbook *orderbook_r, Orderbook *orderbook_t, float open_threshold, float stop_loss_threshold) {
    if (ref_spread.warmed_up() && target_spread.warmed_up() && orderbook_r->TimeStamp != r_sec_timestamp && orderbook_t->TimeStamp != t_sec_timestamp) {
        float spread_stamp_up = 0.0;
        float spread_stamp_down = 0.0;
        float spread_stamp = 0.0;
        r_sec_timestamp = orderbook_r->TimeStamp;
        t_sec_timestamp = orderbook_t->TimeStamp;
        auto [ref_level_ask, target_level_ask] = get_level_asks(orderbook_r, orderbook_t);
        auto [ref_level_bid, target_level_bid] = get_level_bids(orderbook_r, orderbook_t);
        float ref_mid_price = (ref_level_ask + ref_level_bid) / 2;
        float target_mid_price = (target_level_ask + target_level_bid) / 2;
        UNUSED(ref_mid_price);
        UNUSED(target_mid_price);
        if (ref_timestamp != orderbook_r->TimeStamp && target_timestamp != orderbook_t->TimeStamp) {
            ref_timestamp = orderbook_r->TimeStamp;
            target_timestamp = orderbook_t->TimeStamp;
            cointegration_check = false;
            timestamp_check = true;
        } else {
            timestamp_check = false;
        }
        if (position == 0 && !cointegration_check && timestamp_check) {
            std::cout << "test cointegration " << std::endl;
            auto table = cointegration_test();
            if (table.model > 0 && table.model < 4 && table.w1 * table.w2 < 0) {
                cointegration_check = true;
            }
            if (position == 0 && cointegration_check) {
                if (table.w1 < 0 && table.w2 > 0) {
                    spread_stamp_up = table.w1 * log(ref_level_ask) + table.w2 * log(target_level_bid);
                    spread_stamp_down = table.w1 * log(ref_level_bid) + table.w2 * log(target_level_ask);
                } else if (table.w1 > 0 && table.w2 < 0) {
                    spread_stamp_down = table.w1 * log(ref_level_ask) + table.w2 * log(target_level_bid);
                    spread_stamp_up = table.w1 * log(ref_level_bid) + table.w2 * log(target_level_ask);
                }
                if (spread_stamp_up > open_threshold * table.std + table.mu && spread_stamp_up < table.mu + table.std * stop_loss_threshold) {
                    position = -1;
                    std::cout << "up open threshold" << std::endl;
                    if (table.w1 < 0 && table.w2 > 0) {
                        std::cout << "open Quotes setting" << std::endl;
                        open_Quotes_setting(ref_level_ask, target_level_bid);
                    } else if (table.w1 > 0 && table.w2 < 0) {
                        std::cout << "open Quotes setting" << std::endl;
                        open_Quotes_setting(ref_level_bid, target_level_ask);
                    }
                } else if (spread_stamp_down < table.mu - open_threshold * table.std && spread_stamp_down > table.mu - table.std * stop_loss_threshold) {
                    position = 1;
                    std::cout << "down open threshold" << std::endl;
                    if (table.w1 < 0 && table.w2 > 0) {
                        std::cout << "open Quotes setting" << std::endl;
                        open_Quotes_setting(ref_level_bid, target_level_ask);
                    } else if (table.w1 > 0 && table.w2 < 0) {
                        std::cout << "open Quotes setting" << std::endl;
                        open_Quotes_setting(ref_level_ask, target_level_bid);
                    }
                } else if (position != 0) {
                    if (position == -1) {
                        if (ref_size < 0 && target_size > 0) {
                            spread_stamp = table.w1 * log(ref_level_bid) + table.w2 * log(target_level_ask);
                        } else if (ref_size > 0 && target_size < 0) {
                            spread_stamp = table.w1 * log(ref_level_ask) + table.w2 * log(target_level_bid);
                        }
                        if (spread_stamp <= table.mu) {
                            cointegration_check = false;
                            if (ref_size < 0 && target_size > 0) {
                                std::cout << "close Quotes setting" << std::endl;
                                close_Quotes_setting(ref_level_bid, target_level_ask);
                            } else if (ref_size > 0 && target_size < 0) {
                                std::cout << "close Quotes setting" << std::endl;
                                close_Quotes_setting(ref_level_ask, target_level_bid);
                            }
                        } else if (spread_stamp >= table.mu + table.std * stop_loss_threshold) {
                            cointegration_check = false;
                            if (ref_size < 0 && target_size > 0) {
                                std::cout << "close Quotes setting" << std::endl;
                                close_Quotes_setting(ref_level_bid, target_level_ask);
                            } else if (ref_size > 0 && target_size < 0) {
                                std::cout << "close Quotes setting" << std::endl;
                                close_Quotes_setting(ref_level_ask, target_level_bid);
                            }
                        }
                    } else if (position == 1) {
                        if (ref_size < 0 && target_size > 0) {
                            spread_stamp = table.w1 * log(ref_level_ask) + table.w2 * log(target_level_bid);
                        } else if (ref_size > 0 && target_size < 0) {
                            spread_stamp = table.w1 * log(ref_level_bid) + table.w2 * log(target_level_ask);
                        }
                        if (spread_stamp >= table.mu) {
                            cointegration_check = false;
                            if (ref_size < 0 && target_size > 0) {
                                std::cout << "close Quotes setting" << std::endl;
                                close_Quotes_setting(ref_level_ask, target_level_bid);
                            } else if (ref_size > 0 && target_size < 0) {
                                std::cout << "close Quotes setting" << std::endl;
                                close_Quotes_setting(ref_level_bid, target_level_ask);
                            }
                        } else if (spread_stamp <= table.mu + table.std * stop_loss_threshold) {
                            cointegration_check = false;
                            if (ref_size < 0 && target_size > 0) {
                                std::cout << "close Quotes setting" << std::endl;
                                close_Quotes_setting(ref_level_ask, target_level_bid);
                            } else if (ref_size > 0 && target_size < 0) {
                                std::cout << "close Quotes setting" << std::endl;
                                close_Quotes_setting(ref_level_bid, target_level_ask);
                            }
                        }
                    }
                }
            }
        }
        return {SQ_R, SQ_T};
    }
}