__all__ = ["WikiDatabase"]
import copy
import json
import sqlite3

import pandas as pd
import redis

# import timeList
from .properties import properties


class WikiDatabase:
    conn = None

    def __init__(self, run_params):
        self.run_params = run_params
        try:
            self.properties_dict = properties()
            self.conn = sqlite3.connect(self.run_params.db)
            self.redis_connection = redis.Redis(
                host="morain_redis",
                port=6379,
                charset="utf-8",
                decode_responses=True,
            )
        except Exception as e:
            print(e)
            exit(-1)

    def exit_procedure(self):
        self.conn.close()
        return None

    def get_knowledge(self, entity_string):
        """
        Given an entity string, return a complete knowledge dictionary
        """
        knowledge_dict = self.redis_connection.hgetall(entity_string)
        if knowledge_dict:
            return knowledge_dict
        self.conn = sqlite3.connect(self.run_params.db)
        # Get entity
        entity = self.get_entity_by_label(entity_string)
        if entity is None:
            return None

        # Get associations
        associations = self.get_entity_associations(entity[0])

        # Format associations
        knowledge_dict = self.format_associations(entity, associations)
        self.redis_connection.hset(entity_string, mapping=knowledge_dict)

        return knowledge_dict

    def get_associated_entities(self, entity_string):
        self.conn = sqlite3.connect(self.run_params.db)
        # Get entity
        entity = self.get_entity_by_label(entity_string)
        if entity is None:
            return None

        # Get associated entities
        sql = f"""
                SELECT relations.entity_label, relations.description, relations.property_label, Entities.label as related_entity_label
                FROM Entities
                JOIN (
                SELECT e.entity_id, e.label AS entity_label, e.description AS description, p.property_id, p.label AS property_label, pr.related_entity_id
                FROM Properties_relations as pr
                LEFT JOIN Entities as e
                ON pr.entity_id = e.entity_id
                LEFT JOIN Properties as p
                ON pr.property_id = p.property_id
                WHERE pr.related_entity_id = "{entity[0]}"
                )
                AS relations
                ON relations.related_entity_id = Entities.entity_id
                LIMIT 10;
        """
        associations = pd.read_sql_query(sql, self.conn)
        knowledge_dict = {
            "_id": entity[0],
            "label": entity[1],
            "description": entity[2],
            "associated_entities": [],
        }
        # Add associations
        for row in associations.itertuples(index=False):
            associated_entity = {"label": row[0], "description": row[1]}
            associated_entity[row[2]] = row[3]
            knowledge_dict["associated_entities"].append(associated_entity)

        return knowledge_dict

    def get_knowledge_by_id(self, entity_id):
        entity = self.get_entity_by_id(entity_id, all_knowledge=True)
        associations = self.get_entity_associations(entity_id)
        knowledge_dict = self.format_associations(entity, associations)
        return knowledge_dict

    @staticmethod
    def get_table_name(entity_label):
        """
        @:param self:
        @:param label of the entity for which you need to
        :rtype: string
        @:return name of the table where entity is stored
        """
        if entity_label[0].isalpha():
            return entity_label[0].lower()
        else:
            return "not_alpha"

    @staticmethod
    def clean_relations(relations_df):
        """
        :param relations_df: Pandas dataframe of all the relations returned by database
        :return: list of all relations
        :rtype: List
        """
        relations = []
        for index, row in relations_df.iterrows():
            relations.append([row["property_id"], row["related_entity_id"]])
        return relations

    @classmethod
    def get_property_label(self, relations_id):
        """
        :param property_id: id of the property to look for
        :type property_id:
        :return: label of the given id, None if the id does not exist
        :rtype:
        """
        return self.property_dict[relations_id]

    @staticmethod
    def remove_quotations(label):
        index = 0
        while index < len(label):
            if label[index] == "'":
                label = label[:index] + "'" + label[index:]
                index += 1
            elif label[index] == '"':
                label = label[: index - 1] + "'" + "'" + label[index:]
                index += 1
            index += 1

        return label

    # TODO: Test the return value
    # TODO: change to check in the entities table
    def get_similar_entities_label(self, label):
        """
        Returns a list of entities id where the label matches the entity_label name
        :param self:
        :type self:
        :param label:
        :type label:
        :return: list of all the entities id and label where the label matched
        :rtype: Array containing all similar entities
        """
        entities_id = None
        try:
            entities_id = pd.read_sql_query(
                'SELECT * FROM Entities WHERE label  LIKE "%{}%";'.format(
                    self.remove_quotations(label)
                ),
                self.conn,
            )
        except Exception as e:
            print(e)
            self.exit_procedure()
        return entities_id.values.tolist()

    def get_entities_by_label_extensive(self, label):
        """
        Returns a list of entities id where the label matches the entity_label name
        :param self:
        :type self:
        :param label:
        :type label:
        :return: list of all the entities id and label where the label matched
        :rtype: Array containing all similar entities
        """
        table_name = self.get_table_name(label)
        entities_id = None
        try:
            entities_id = pd.read_sql_query(
                'SELECT * FROM Entities_{} WHERE label  LIKE "{}%";'.format(
                    table_name, self.remove_quotations(label)
                ),
                self.conn,
            )
            return entities_id.values.tolist()
        except Exception as e:
            print(e)
            self.exit_procedure()

    def get_entity_by_label(self, label):
        """
        Search for the id, label and description of an specific identity
        :param label: string of the label to look for
        :type label: string
        :return: id, label and description of the label
        :rtype: pandas.core.series.Series
        """
        if label is None:
            return None
        entity_id = None
        table_name = self.get_table_name(label)
        try:
            entity_id = pd.read_sql_query(
                'SELECT * FROM Entities_{} WHERE label =  "{}";'.format(
                    table_name, self.remove_quotations(label)
                ),
                self.conn,
            )
            if entity_id.empty:
                entities = self.get_entities_by_label_extensive(label)
            else:
                entities = entity_id.values.tolist()
                # Return the first entity
                entities = self._sort_entities(entities)
            return entities[0]
        except Exception as e:
            self.exit_procedure()

    def _sort_entities(self, entities):
        """
        Given a list of entities, sort the list where the lowest id number is first.

        We are assuming that the most likely entity is the one with the lowest id number
        in the knowledge base.
        """
        # Chops off the first letter Q and casts the string as a number
        return sorted(entities, key=lambda x: int(x[0][1:]))

    def get_entity_associations(self, entity_id):
        """
        Search for all properties of the entity
        Return all the
        :param entity_id: id of a entity
        :type entity_id: str
        :return: data frame with a string of the relations
        :rtype: pandas.core.series.Series
        """
        entity_properties = None
        try:
            entity_properties = pd.read_sql_query(
                'SELECT * FROM Properties_relations WHERE entity_id = "{}";'.format(
                    entity_id
                ),
                self.conn,
            )
        except Exception as e:
            print(e)
            self.exit_procedure()
        if entity_properties.empty:
            return None
        return self.clean_relations(entity_properties)

    def get_entity_by_id(self, entity_id, all_knowledge=False):

        relation_label = None
        try:
            relation_label = pd.read_sql_query(
                'SELECT label, description FROM Entities WHERE entity_id = "{}"'.format(
                    entity_id
                ),
                self.conn,
            )
        except Exception as e:
            print(e)
            self.exit_procedure()
        if relation_label.empty:
            return None
        if all_knowledge:
            entity = (entity_id, relation_label.label[0], relation_label.description[0])
            return entity
        return relation_label.iloc[0]["label"]

    def get_property_string(self, property_id, related_entity_id):
        """
        :param property_id:
        :type property_id: string
        :param related_entity_id:
        :type related_entity_id: string
        :return:
        :rtype: string describing the relationship
        """
        entity_label = None
        related_entity_label = None
        property_name = None
        try:
            # entity_label = get_entity_label(entity_id)
            related_entity_label = self.get_entity_by_id(related_entity_id)
            property_name = self.properties_dict[property_id]
            # print(related_entity_label, property_name)
        except Exception as e:
            print(e)
            self.exit_procedure()
        return property_name, related_entity_label

    def format_associations(self, entity, associations):
        """
        Given an entity and it's associations, format them as a dictionary

        Example:
            {
                'label' : 'Stephen Curry',
                'description' : 'American Basketball player',
                'plays for' :  'Golden State Warriors'
            {
        """
        if not entity:
            return None
        entity_associations_dict = {
            "id": entity[0],
            "label": entity[1],
            "description": entity[2],
        }
        # Remove all None values from list
        if not associations:
            return None
        for property_id, related_entity_id in associations:
            property_name, related_entity_label = self.get_property_string(
                property_id, related_entity_id
            )
            if related_entity_label:
                entity_associations_dict[property_name] = related_entity_label
        return entity_associations_dict
