-- Get an entity and all of it's relations
-- SELECT e.label AS entity_label, p.label AS property_label, pr.related_entity_id as related_entity_id 
-- FROM Entities AS e, Properties_relations AS pr, Properties AS p
-- WHERE e.entity_id = pr.entity_id
-- LIMIT 10;
SELECT relations.*, Entities.label as related_entity_label
FROM Entities
JOIN (
SELECT e.entity_id, e.label, p.property_id, p.label, pr.related_entity_id
FROM Properties_relations as pr
LEFT JOIN Entities as e
ON pr.entity_id = e.entity_id
LEFT JOIN Properties as p
ON pr.property_id = p.property_id
WHERE pr.entity_id = "Q31"
)
AS relations
ON relations.related_entity_id = Entities.entity_id
LIMIT 10;
-- SELECT relations.*, Entities.label as related_entity_label
-- FROM Entities
-- JOIN (SELECT e.label AS entity_label, p.label AS property_label, pr.related_entity_id as related_entity_id 
-- FROM Entities AS e, Properties_relations AS pr, Properties AS p
-- WHERE e.entity_id = pr.entity_id)
-- AS relations 
-- ON relations.related_entity_id = Entities.entity_id
-- LIMIT 10
-- ;
